# TCM-RAG

TCM-RAG 是一个面向中医知识问答的检索增强生成系统。项目将本地中医文献解析、语义切分、混合检索、重排序、对话记忆和医疗安全路由组合成一个可运行的问答 Agent，并提供命令行和 Web 两种使用方式。

> 本项目仅用于中医知识检索、学习与问答实验，不构成诊断或处方建议。涉及急症、特殊人群、用药剂量或明确诊断时，应以线下正规医疗机构评估为准。


<img width="1434" height="1044" alt="image" src="https://github.com/user-attachments/assets/157c525e-99f5-409a-a7c8-110961cb4b0c" />

## 功能特性

- 支持 PDF、TXT、Markdown、JSON 知识源解析
- 使用 BGE 向量模型构建 FAISS-HNSW 稠密检索索引
- 结合 BM25 稀疏检索、RRF 融合与 BGE Reranker 重排序
- 内置短期对话记忆、摘要压缩和长期记忆检索
- 面向医疗问答的安全路由：追问、紧急就医提示、答案安全提示
- 支持 RAG 质量评估与安全评测
- 提供命令行交互与本地 Web 控制台

## 项目结构

```text
.
├── config/              # 配置项
├── core/                # Agent、检索、记忆、安全与评估核心逻辑
├── data/
│   ├── raw/             # 原始知识源
│   ├── processed/       # 语义切分缓存，默认不提交
│   └── memory_store/    # FAISS 索引与元数据，默认不提交
├── eval/                # RAG 与安全评测数据集
├── parsers/             # PDF/TXT/Markdown/JSON 解析器
├── utils/               # 文本切分等工具
├── web/                 # 本地 Web 服务和前端页面
├── main.py              # 命令行入口
└── requirements.txt     # Python 依赖
```

## 环境要求

- Python 3.10+
- 推荐使用 CUDA 环境运行 BGE embedding 和 reranker
- 可访问 DeepSeek/OpenAI 兼容接口的 API Key

如果只使用 CPU，需要根据本机环境调整 `core/retriever.py` 中模型加载参数，或安装对应的 CPU 版依赖。

## 快速开始

1. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. 配置环境变量：

```bash
copy .env.example .env
```

然后编辑 `.env`：

```env
LLM_API_KEY=API密钥
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-chat
```

3. 放入知识源文件：

将 PDF、TXT、Markdown 或 JSON 文件放入 `data/raw/`。首次启动或知识源更新后，系统会自动构建或重建索引。

4. 启动命令行问答：

```bash
python main.py
```

命令行内支持：

- `new`：开启新对话
- `rebuild`：重建知识库索引
- `eval`：运行 RAG 批量评估
- `safetyeval`：运行安全评测
- `quit`：退出

## Web 控制台

```bash
python web/server.py --host 127.0.0.1 --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

Web 页面支持问答、新对话、重建知识库、RAG 评估、安全评测，并展示置信度、检索轮次、安全路由和引用来源。

## 评估

RAG 评估数据位于：

```text
eval/rag_eval_set.json
```

安全评测数据位于：

```text
eval/safety_eval_set.json
```

运行方式：

```bash
python main.py
# 输入 eval 或 safetyeval
```

生成的评估报告默认写入 `eval/rag_eval_report.json`，该文件已加入 `.gitignore`。

## 配置说明

主要配置位于 `config/settings.py`，并可通过 `.env` 覆盖：

- `LLM_API_KEY`：LLM API Key
- `LLM_BASE_URL`：OpenAI 兼容接口地址
- `LLM_MODEL_NAME`：聊天模型名称
- `EMBEDDING_MODEL_PATH`：向量模型路径
- `RERANK_MODEL_PATH`：重排序模型路径
- `FAISS_INDEX_PATH`：FAISS 索引保存位置
- `TOP_K_DENSE` / `TOP_K_SPARSE` / `TOP_K_RERANK`：召回与重排序数量
- `CONFIDENCE_THRESHOLD`：Agent 回答置信度阈值

