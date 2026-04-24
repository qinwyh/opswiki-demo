# OpsWiki – 本地代码仓库理解与问答系统 (DeepWiki MVP)

一个面向本地代码仓库的 DeepWiki 风格 MVP 系统，支持结构化代码解析、混合检索、流式问答、引用溯源和 Mermaid 图生成。

## 快速开始

### 1. 安装 uv（如未安装）

```bash
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 创建环境并安装依赖

```bash
cd opswiki-demo
uv venv
uv sync
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

必需配置：
- `QWEN_API_KEY`：阿里云百炼 API Key（或 `DASHSCOPE_API_KEY`）

### 4. 启动服务

```bash
uv run python main.py
```

打开浏览器访问 http://localhost:8000

## 使用方式

1. 在左侧输入本地代码仓库的**绝对路径**
2. 点击"导入并建立索引"
3. 等待索引完成（取决于仓库大小）
4. 在主区域输入问题，支持：
   - 代码定位：`parse_python 函数做了什么？`
   - 文档问答：`README 中描述了哪些功能？`
   - 复杂推理：`请生成核心模块的调用时序图`

## 项目结构

```
opswiki-demo/
├── app/
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── schemas.py         # Pydantic 数据模型
│   ├── ingest.py          # 仓库扫描、AST/Markdown 解析、结构化分块
│   ├── indexing.py         # FAISS 向量索引、BM25、SQLite 元数据存储
│   ├── retrieval.py        # 查询理解、多路召回、RRF融合、重排、图扩展
│   ├── qa.py              # 问答链、复杂问题分解、Mermaid生成
│   ├── api.py             # FastAPI 路由
│   └── frontend/
│       ├── templates/
│       │   └── index.html  # 主页面
│       └── static/
│           ├── style.css   # 样式
│           └── app.js      # 前端交互逻辑
├── data/                   # 索引、元数据存储（自动创建）
├── main.py                # 入口
├── pyproject.toml
├── .env.example
└── README.md
```

## 架构设计

### 核心链路
```
用户问题 → 问题分类 → 查询翻译(多变体) → 多路召回(Dense+Sparse+Exact) → RRF融合 → 重排 → 图扩展 → LLM生成(流式) → 结构化输出(answer+citations+mermaid)
```

### 分块策略
- **Python**: AST 结构化分块 (file_summary / class / function / method)，提取签名、docstring、imports、calls
- **Markdown**: 按标题层级分块，保留 heading path

### 检索设计
- Dense: FAISS (内积, L2归一化 = 余弦相似度)
- Sparse: BM25
- Exact: SQLite 符号名/路径精确匹配
- 融合: Reciprocal Rank Fusion (RRF)
- 重排: 启发式特征 (可替换为 Cross-Encoder)
- 图扩展: SQLite 边表 1-hop 邻域补充

### 可扩展点
- **Embedding**: 抽象为 `EmbeddingProvider` 接口，可切换本地模型
- **Reranker**: 抽象为 `Reranker` 接口，可接入 Cross-Encoder
- **LLM**: 通过 LangChain ChatOpenAI 接入，可替换任意兼容端点
- **解析器**: 可扩展支持 .ts/.java/.go 等语言
- **索引**: 可替换为 Milvus 等向量数据库

## API 文档

启动后访问 http://localhost:8000/docs 查看 Swagger 文档。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/import` | POST | 导入并索引仓库 `{"path": "/abs/path"}` |
| `/api/repo/{id}` | GET | 获取仓库状态 |
| `/api/repo/{id}/files` | GET | 获取文件树 |
| `/api/ask` | POST | 提问 (SSE流式) `{"repo_id": "...", "question": "..."}` |

## 技术栈

- Python 3.11+ / FastAPI / Jinja2
- LangChain + Qwen (阿里云百炼 OpenAI兼容)
- FAISS (向量检索) + BM25 (稀疏检索)
- SQLite (元数据 + 代码关系图)
- 原生 JS + CSS (无 Node 依赖)
- Mermaid.js CDN (图表渲染)
