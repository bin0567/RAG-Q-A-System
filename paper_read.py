import os
import json
import tempfile
import shutil
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


# ==================== 全局配置参数 ====================
EMBEDDING_MODEL = "qwen3-embedding:0.6b"  # 嵌入模型名称
LLM_MODEL = "deepseek-r1:8b"  # LLM模型名称
LLM_TEMPERATURE = 0.1  # 模型温度，控制随机性
LLM_NUM_CTX = 4096  # 上下文窗口大小
LLM_NUM_GPU = 999  # 使用的GPU数量
CHUNK_SIZE = 1000  # 文档分块大小
CHUNK_OVERLAP = 100  # 分块重叠大小
SEARCH_K = 5  # 检索返回的相关文档数量
SERVER_HOST = "0.0.0.0"  # 服务器监听地址
SERVER_PORT = 8000  # 服务器端口号
# ====================================================


# 系统提示词模板，定义AI助手的角色和回答规范
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
    """学术论文RAG问答系统核心类"""

    def __init__(self):
        # 初始化嵌入模型，用于将文档和查询向量化
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, keep_alive=-1)
        # 初始化LLM模型，用于生成回答
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            num_ctx=LLM_NUM_CTX,
            num_gpu=LLM_NUM_GPU,
            keep_alive=-1,
        )
        self.vectorstore = None  # FAISS向量数据库实例
        self.retriever = None  # 检索器实例
        self.conversation_chain = None  # 对话链实例
        self.chat_history = ChatMessageHistory()  # 聊天历史记录
        self.temp_dir = None  # 临时目录，用于存储上传的文件

    def _ensure_temp_dir(self):
        """确保临时目录存在"""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()

    def load_document(self, file_path: str, original_name: str):
        """加载单个文档，支持PDF和Word格式"""
        ext = os.path.splitext(original_name)[1].lower()
        docs = []

        try:
            if ext == '.pdf':
                # 优先使用PyPDFLoader，失败则使用UnstructuredPDFLoader
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                except Exception:
                    loader = UnstructuredPDFLoader(file_path)
                    docs = loader.load()
            elif ext in ['.docx', '.doc']:
                # 优先使用Docx2txtLoader，失败则使用UnstructuredWordDocumentLoader
                try:
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                except Exception:
                    loader = UnstructuredWordDocumentLoader(file_path)
                    docs = loader.load()

            # 为每个文档添加来源文件名元数据
            for doc in docs:
                doc.metadata['source'] = original_name

            print(f"解析成功: {original_name} - {len(docs)} 页/段")
        except Exception as e:
            print(f"解析失败 {original_name}: {e}")

        return docs

    async def add_documents(self, files: List[UploadFile]):
        """添加文档到系统中：保存文件、解析内容、构建向量库"""
        self._ensure_temp_dir()
        new_files = []
        total_docs = []

        # 遍历上传的文件
        for file in files:
            # 保存文件到临时目录
            file_path = os.path.join(self.temp_dir, file.filename)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # 加载文档内容
            docs = self.load_document(file_path, file.filename)
            if docs:
                new_files.append(file.filename)
                total_docs.extend(docs)

        if not new_files:
            return {'success': False}

        # 文档分块处理，将长文档分割成较小的段落
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        splits = text_splitter.split_documents(total_docs)
        print(f"文档分块完成: {len(splits)} 个块")

        # 构建FAISS向量数据库
        self.vectorstore = FAISS.from_documents(documents=splits, embedding=self.embeddings)
        # 创建检索器，指定返回5个相关文档
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})

        # 创建问答链：结合检索器和LLM
        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
        )
        self.conversation_chain = create_retrieval_chain(self.retriever, question_answer_chain)

        return {'success': True, 'added': new_files, 'total': new_files}

    def ask_stream(self, question: str):
        """流式问答：边生成回答边返回，支持分离思考过程和最终回答"""
        # 将聊天历史转换为消息格式
        history_messages = []
        for msg in self.chat_history.messages:
            if isinstance(msg, HumanMessage):
                history_messages.append(("human", msg.content))
            elif isinstance(msg, AIMessage):
                history_messages.append(("ai", msg.content))

        full_answer = ""

        if self.conversation_chain:
            # 流式获取响应
            response = self.conversation_chain.stream({
                "input": question,
                "chat_history": history_messages
            })

            in_thinking = False  # 是否处于思考标签内
            thinking = ""  # 思考内容累积

            for event in response:
                if "answer" in event:
                    content = event["answer"]
                    full_answer += content

                    # 解析DeepSeek模型的思考标签
                    if "<think>" in content:
                        # 进入思考标签
                        in_thinking = True
                        thinking += content.split("<think>")[1]
                        yield json.dumps({"type": "thinking", "content": thinking}, ensure_ascii=False) + "\n"
                    elif "</think>" in content:
                        # 退出思考标签
                        in_thinking = False
                        parts = content.split("</think>")
                        thinking += parts[0]
                        yield json.dumps({"type": "thinking_done", "content": thinking}, ensure_ascii=False) + "\n"
                        if len(parts) > 1 and parts[1]:
                            yield json.dumps({"type": "answer", "content": parts[1]}, ensure_ascii=False) + "\n"
                    elif in_thinking:
                        # 持续累积思考内容
                        thinking += content
                        yield json.dumps({"type": "thinking", "content": content}, ensure_ascii=False) + "\n"
                    else:
                        # 普通回答内容
                        yield json.dumps({"type": "answer", "content": content}, ensure_ascii=False) + "\n"

        # 将完整回答添加到聊天历史
        self.chat_history.add_ai_message(full_answer)
        yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n"

    def clear_history(self):
        """清空聊天历史"""
        self.chat_history.clear()
        print("聊天历史已清空。")

    def reset(self):
        """重置整个系统：清空向量库、会话链和临时文件"""
        self.vectorstore = None
        self.conversation_chain = None
        self.chat_history.clear()
        # 清理临时目录
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.temp_dir = None
        print("已重置会话")


# 创建全局RAG实例
rag = PaperRAG()
# 创建FastAPI应用实例
app = FastAPI()


class QuestionRequest(BaseModel):
    """用户问题请求模型"""
    question: str


@app.get("/")
async def index():
    """首页路由，返回聊天界面HTML"""
    with open("chat.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """文件上传API"""
    result = await rag.add_documents(files)
    if result['success']:
        return {"success": True, "added": result['added'], "total": result['total']}
    else:
        return {"success": False, "message": "解析失败，请重试"}


@app.post("/api/chat/stream")
async def chat_stream(req: QuestionRequest):
    """流式问答API"""
    return StreamingResponse(rag.ask_stream(req.question), media_type="text/event-stream")


@app.post("/api/clear")
async def clear_history():
    """清空聊天历史API"""
    rag.clear_history()
    return {"status": "ok"}


@app.post("/api/reset")
async def reset():
    """重置系统API"""
    rag.reset()
    return {"status": "ok"}


if __name__ == "__main__":
    print("=" * 60)
    print("文档智能问答助手")
    print("=" * 60)
    print(f"\n启动 Web 服务... 访问地址: http://localhost:{SERVER_PORT}\n")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
