# 文档智能问答助手

一个基于本地大模型的学术文献智能问答系统，支持PDF和Word文档上传，能够基于文档内容进行精准问答，并提供结构化的回答格式和聊天历史管理功能。

## 功能特性

### 核心功能
- 📄 **多格式文档支持**：支持PDF、DOCX、DOC格式文档上传和解析
- 🧠 **本地模型推理**：基于Ollama运行本地大模型，数据隐私有保障
- 🔍 **精准问答**：基于RAG（检索增强生成）技术，仅从上传文档中提取信息回答问题
- 📝 **结构化回答**：自动使用二级标题、加粗、列表等格式组织回答内容
- 💬 **聊天历史管理**：支持查看、跳转和清空聊天历史记录
- ⚡ **流式响应**：实时返回回答内容，包含思考过程展示
- 🔄 **会话重置**：支持清空所有上传文档和聊天记录，重新开始

### 技术特性
- 文档分块处理，优化长文档检索效率
- 向量存储使用FAISS，快速相似性检索
- 支持追加上传多个文档
- 响应式UI设计，适配不同屏幕尺寸
- 实时上传状态提示
- 平滑的动画和交互效果

## 技术栈

### 后端
- Python 3.8+
- FastAPI：高性能API框架
- LangChain：构建LLM应用的框架
- FAISS：高效向量存储和检索
- Ollama：本地大模型运行环境
- PyPDFLoader/UnstructuredPDFLoader：PDF解析
- Docx2txtLoader/UnstructuredWordDocumentLoader：Word文档解析

### 前端
- HTML5/CSS3
- JavaScript (ES6+)
- 原生CSS动画和交互
- 流式响应处理

## 安装部署

### 前置条件
1. 安装Python 3.8+
2. 安装Ollama并下载所需模型：
   ```bash
   # 安装Ollama (Windows版本)
   # 访问 https://ollama.ai/download 下载Windows安装包
   # 安装完成后，在命令提示符或PowerShell中执行以下命令
   
   # 下载嵌入模型
   ollama pull qwen3-embedding:0.6b
   
   # 下载对话模型
   ollama pull deepseek-r1:8b
   ```

### 安装步骤

1. **准备项目文件**
   - 将 `paper_read.py` 和 `chat.html` 两个文件放置在同一文件夹下
   - 建议创建一个专门的项目文件夹，例如 `C:\document-qa-assistant`

2. **创建并激活虚拟环境**
   ```bash
   # 打开命令提示符或PowerShell，进入项目文件夹
   cd C:\document-qa-assistant
   
   # 创建虚拟环境
   python -m venv venv
   
   # 激活虚拟环境
   venv\Scripts\activate
   ```

3. **安装依赖包**
   ```bash
   # 使用国内镜像源安装依赖，速度更快
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple fastapi uvicorn langchain langchain-community langchain-ollama faiss-cpu pypdf docx2txt unstructured python-multipart
   ```

   或者创建一个 `requirements.txt` 文件，内容如下：
   ```
   fastapi>=0.104.1
   uvicorn>=0.24.0
   langchain>=0.1.0
   langchain-community>=0.0.19
   langchain-ollama>=0.1.0
   faiss-cpu>=1.7.4
   pypdf>=3.17.0
   docx2txt>=0.8
   unstructured>=0.10.30
   python-multipart>=0.0.6
   ```
   
   然后使用以下命令安装：
   ```bash
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
   ```

4. **启动应用**
   ```bash
   python paper_read.py
   ```

5. **访问系统**
   - 启动成功后，会看到提示信息：`访问地址: http://localhost:8000`
   - 打开浏览器（推荐Chrome、Edge或Firefox）访问：http://localhost:8000

## 使用指南

### 基础使用

1. **上传文档**
   - 点击界面上的"上传文档"按钮
   - 选择一个或多个PDF/DOCX/DOC文件
   - 等待文档解析完成（会显示上传状态提示）
   - 成功上传后，会显示当前已上传的文件列表

2. **提问**
   - 在输入框中输入关于上传文档的问题
   - 按Enter键或点击发送按钮提交问题
   - 等待系统生成回答（会显示思考过程和实时生成的回答）

3. **查看回答**
   - 回答会自动以结构化格式展示（二级标题、加粗、列表等）
   - 关键概念和结论会自动加粗标注
   - 回答末尾会标注来源文档

### 高级功能

#### 聊天历史管理
- 右侧边栏显示所有历史提问记录
- 点击历史记录可快速跳转到对应的对话位置
- 点击"清空历史"按钮可删除所有历史记录

#### 会话管理
- 点击"重置"按钮可清空所有上传的文档和聊天记录
- 点击"清空历史"仅删除聊天记录，保留上传的文档

## 系统架构

### 后端流程
1. **文档处理流程**
   - 上传文件保存到临时目录
   - 根据文件类型选择合适的加载器解析文档
   - 将文档分割为固定大小的块（chunk_size=500，重叠50字符）
   - 使用嵌入模型生成文档块的向量表示
   - 将向量存储到FAISS向量数据库

2. **问答流程**
   - 接收用户问题，生成问题向量
   - 从FAISS中检索最相关的5个文档块
   - 将检索到的上下文和问题一起发送给LLM
   - 流式返回LLM生成的回答
   - 记录聊天历史

### 关键组件
- **PaperRAG类**：核心RAG逻辑实现
  - `load_document()`: 文档解析
  - `add_documents()`: 添加和处理上传的文档
  - `_rebuild_vectorstore()`: 重建向量库
  - `ask_stream()`: 流式问答接口
  - `clear_history()`: 清空聊天历史
  - `reset()`: 重置整个会话

- **API端点**
  - `GET /`: 提供前端页面
  - `POST /api/upload`: 文档上传接口
  - `POST /api/chat/stream`: 流式问答接口
  - `POST /api/clear`: 清空聊天历史
  - `POST /api/reset`: 重置会话

## 自定义配置

### 修改模型配置
在`paper_read.py`中修改PaperRAG初始化参数：
```python
rag = PaperRAG(
    embedding_model="qwen3-embedding:0.6b",  # 嵌入模型
    llm_model="deepseek-r1:8b"               # 对话模型
)
```

支持的模型配置：
- `embedding_model`: 任何Ollama支持的嵌入模型
- `llm_model`: 任何Ollama支持的对话模型
- `temperature`: 模型生成温度（默认0.1，越低越精准）
- `num_ctx`: 上下文窗口大小（默认4096）
- `k`: 检索的文档块数量（默认5）

### 修改文档分块配置
在`_rebuild_vectorstore()`方法中修改：
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每个块的大小
    chunk_overlap=50,     # 块之间的重叠字符数
    length_function=len,
)
```

### 修改回答格式
修改`SYSTEM_PROMPT`常量可以调整回答的格式要求和行为规则。

## 故障排除

### 常见问题

1. **模型加载失败**
   - 确保Ollama服务正在运行：在任务管理器中查看Ollama进程
   - 确认模型已正确下载：在命令提示符中执行 `ollama list`
   - 检查模型名称是否正确
   - 尝试重启Ollama服务

2. **文档解析失败**
   - 确保文件未损坏
   - 对于复杂的PDF，系统会自动切换到UnstructuredPDFLoader
   - 检查文件权限
   - 尝试将文件另存为新的PDF或Word文档

3. **回答为空或不相关**
   - 确认问题与上传的文档内容相关
   - 尝试调整检索块数量（k值）
   - 检查文档是否被正确解析
   - 尝试更具体的问题

4. **性能问题**
   - 降低chunk_size可以提高检索速度
   - 使用更小的模型可以加快响应速度
   - 增加系统内存，特别是处理大文档时
   - 关闭其他占用内存的程序

5. **端口被占用**
   - 如果8000端口被占用，可以在`paper_read.py`的`main()`函数中修改端口号
   - 找到 `uvicorn.run(app, host="0.0.0.0", port=8000)` 这一行，将8000改为其他端口，如8080

6. **依赖安装失败**
   - 确保使用国内镜像源
   - 尝试逐个安装依赖包，定位具体是哪个包安装失败
   - 检查Python版本是否符合要求
   - 确保有足够的磁盘空间

## 界面预览

### 主界面
- 简洁的聊天界面，左侧为对话区域，右侧为历史记录
- 顶部有文档上传和重置按钮
- 底部有消息输入框

### 回答展示
- 结构化的回答格式，包含二级标题、加粗文本、列表
- 思考过程展示区域
- 平滑的流式响应动画

## 注意事项

1. **数据安全**：所有文档都存储在本地临时目录，重置会话会自动删除
2. **模型要求**：确保有足够的内存运行所选模型（建议至少16GB RAM）
3. **性能优化**：对于非常大的文档，建议拆分后分批上传
4. **网络要求**：首次运行需要下载模型，确保网络畅通
5. **系统兼容性**：本系统仅在Windows系统上测试通过，其他操作系统可能需要调整配置
6. **文件命名**：建议使用英文文件名，避免使用特殊字符
7. **备份重要文档**：在上传重要文档前，建议先备份
8. **定期清理**：长时间使用后，建议定期重置会话以清理临时文件
9. **模型选择**：可以根据硬件配置选择合适大小的模型，不一定使用最大的模型
10. **问题质量**：提问越具体，回答越准确，避免过于宽泛的问题
