import os
import json
import tempfile
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory


SYSTEM_PROMPT = """你是一个专业的学术文献助手，帮助用户理解和分析学术论文。

回答格式要求：
1. **使用二级标题组织回答**：每一个重要的主题都使用 "## 标题名称" 格式作为二级标题
2. **重要内容加粗**：关键概念、定义、结论等重要内容使用 **加粗** 格式标记
3. **条理清晰**：使用数字列表或项目符号来组织多个要点
4. **来源标注**：在回答末尾统一标注来源文档即可，格式为：(来源：文档名)

回答规则：
- 仅基于提供的文档上下文回答问题。如果上下文不包含相关信息，请明确说明。
- 回答要精确严谨，不要编造或臆测信息。
- 如果不确定或上下文不足，承认局限性而不是猜测。
- 使用与用户提问相同的语言回答。
- 对于专业术语，首次出现时在括号中保留英文原文。

文档上下文：
{context}"""


class PaperRAG:
    def __init__(
        self,
        embedding_model: str = "qwen3-embedding:0.6b",
        llm_model: str = "deepseek-r1:8b"
    ):
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            keep_alive=-1,
        )
        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.1,
            num_ctx=4096,
            num_gpu=999,
            keep_alive=-1,
        )
        self.vectorstore = None
        self.retriever = None
        self.conversation_chain = None
        self.chat_history = ChatMessageHistory()
        self.uploaded_files: List[str] = []
        self.all_documents = []
        self.temp_dir = None

    def _ensure_temp_dir(self):
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()

    def load_document(self, file_path: str, original_name: str):
        ext = os.path.splitext(original_name)[1].lower()
        docs = []
        
        try:
            if ext == '.pdf':
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                except Exception:
                    loader = UnstructuredPDFLoader(file_path)
                    docs = loader.load()
            elif ext in ['.docx', '.doc']:
                try:
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                except Exception:
                    loader = UnstructuredWordDocumentLoader(file_path)
                    docs = loader.load()
            
            for doc in docs:
                doc.metadata['source'] = original_name
            
            print(f"解析成功: {original_name} - {len(docs)} 页/段")
        except Exception as e:
            print(f"解析失败 {original_name}: {e}")
        
        return docs

    def add_documents(self, files: List[UploadFile]):
        self._ensure_temp_dir()
        new_files = []
        new_docs = []
        
        for file in files:
            file_path = os.path.join(self.temp_dir, file.filename)
            with open(file_path, 'wb') as f:
                f.write(file.file.read())
            
            docs = self.load_document(file_path, file.filename)
            if docs:
                new_files.append(file.filename)
                new_docs.extend(docs)
        
        if new_docs:
            self.uploaded_files.extend(new_files)
            self.all_documents.extend(new_docs)
            self._rebuild_vectorstore()
        
        return {
            'added': new_files,
            'total': self.uploaded_files,
            'success': len(new_docs) > 0
        }

    def _rebuild_vectorstore(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_documents(self.all_documents)
        print(f"总文档数: {len(self.all_documents)}, 分块数: {len(chunks)}")
        
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self._setup_conversation()

    def _setup_conversation(self):
        self.vectorstore.search_kwargs = {"k": 5}
        self.retriever = self.vectorstore.as_retriever()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        docs_chain = create_stuff_documents_chain(self.llm, prompt)
        self.conversation_chain = create_retrieval_chain(self.retriever, docs_chain)

    def ask_stream(self, question: str):
        if not self.conversation_chain:
            yield json.dumps({"type": "error", "content": "请先上传文献文件！"}, ensure_ascii=False) + "\n"
            return

        self.chat_history.add_user_message(question)

        full_answer = ""
        thinking = ""
        in_thinking = False

        for chunk in self.conversation_chain.stream(
            {"input": question, "chat_history": self.chat_history.messages}
        ):
            if "answer" in chunk:
                content = chunk["answer"]
                full_answer += content

                if "<think>" in content:
                    in_thinking = True
                    thinking += content.split("<think>")[1]
                    yield json.dumps({"type": "thinking", "content": thinking}, ensure_ascii=False) + "\n"
                elif "</think>" in content:
                    in_thinking = False
                    parts = content.split("</think>")
                    thinking += parts[0]
                    yield json.dumps({"type": "thinking_done", "content": thinking}, ensure_ascii=False) + "\n"
                    if len(parts) > 1 and parts[1]:
                        yield json.dumps({"type": "answer", "content": parts[1]}, ensure_ascii=False) + "\n"
                elif in_thinking:
                    thinking += content
                    yield json.dumps({"type": "thinking", "content": content}, ensure_ascii=False) + "\n"
                else:
                    yield json.dumps({"type": "answer", "content": content}, ensure_ascii=False) + "\n"

        self.chat_history.add_ai_message(full_answer)
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

    def clear_history(self):
        self.chat_history.clear()
        print("聊天历史已清空。")

    def reset(self):
        self.vectorstore = None
        self.conversation_chain = None
        self.chat_history.clear()
        self.uploaded_files = []
        self.all_documents = []
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
        self.temp_dir = None
        print("已重置会话")


rag = PaperRAG()
app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
async def index():
    with open("chat.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    result = rag.add_documents(files)
    
    if result['success']:
        return {
            "success": True,
            "added": result['added'],
            "total": result['total'],
            "message": f"成功添加 {len(result['added'])} 个文件，当前共 {len(result['total'])} 个文件"
        }
    else:
        return {
            "success": False,
            "message": "解析失败，请重试"
        }


@app.post("/api/chat/stream")
async def chat_stream(req: QuestionRequest):
    return StreamingResponse(
        rag.ask_stream(req.question),
        media_type="text/event-stream",
    )


@app.post("/api/clear")
async def clear_history():
    rag.clear_history()
    return {"status": "ok"}


@app.post("/api/reset")
async def reset():
    rag.reset()
    return {"status": "ok"}


def main():
    print("=" * 60)
    print("文档智能问答助手")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("启动 Web 服务...")
    print("访问地址: http://localhost:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
