from dotenv import load_dotenv
import os
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# 加载环境变量
load_dotenv()

# 读取 API 密钥
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 设置环境变量（保证兼容性，确保所有依赖这种方式的库都能正常工作。）
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# 从 PDF 文件提取文本
extracted_data = load_pdf_file(data="data/")

# 过滤文档以仅保留必要的元数据
filter_data = filter_to_minimal_docs(extracted_data)

# 将文档拆分为更小的块
text_chunks = text_split(filter_data)

# 从 HuggingFace 下载嵌入模型
embeddings = download_hugging_face_embeddings()

# 初始化 Pinecone 客户端并连接
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

# 创建 Pinecone 索引并连接
index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# 创建 Pinecone 向量存储
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks, embedding=embeddings, index_name=index_name
)
