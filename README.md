# medical-chatbot-full-stack

## 概述

本项目实现了一个基于 RAG 思想和本地知识库的医疗咨询机器人，交互方式为 Web 页面。

<img src="assets/image-20251029210729119.png" alt="image-20251029210729119" style="zoom:50%;" />

## 技术栈

- 使用 LangChain 框架调用“gemini-2.5-flash”模型，搭建智能体。
- 使用 RAG 思想及 Pinecone 数据库构建本地知识库。
- 使用 Flask 框架及 Jinja2 模板构建前端和后端。
- 使用 GitHub Action 进行 CI/CD，将项目部署在 AWS 云服务中。

## 项目仓库

- GitHub: https://github.com/Jerrybaijy/medical-chatbot-full-stack
- Docker Hub: https://hub.docker.com/repository/docker/jerrybaijy/medical-chatbot-full-stack

## RAG

```mermaid
flowchart TD
    A[知识库文档<br>PDF/Word/网页等] --> B[文本分割<br>Chunking]
    B --> C[向量嵌入<br>Embedding Model]
    C --> D[存储到向量数据库<br>Pinecone/Milvus等]
    
    E[用户提问<br>User Prompt] --> F[向量嵌入<br>Same Embedding Model]
    F --> G[相似性搜索<br>Semantic Search]
    D --> G
    G --> H[检索相关文本块<br>as Context]
    H --> I[组合提示词<br>Prompt Engineering]
    I --> J[LLM生成最终答案<br>GPT-4/Claude等]
    J --> K[返回给用户]
```

