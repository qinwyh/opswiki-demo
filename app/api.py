"""
OpsWiki - FastAPI 应用与路由
============================
系统的 HTTP 入口层，负责：
- 提供前端页面（Jinja2 模板渲染）
- REST API 端点（导入仓库、查询状态、流式问答）
- 全局状态管理（MetadataStore、RepoIndex、RetrievalPipeline 的生命周期）

API 端点一览：
- GET  /             → 主页面
- POST /api/import   → 导入并索引仓库
- GET  /api/repo/{id}       → 获取仓库信息
- GET  /api/repo/{id}/files → 获取仓库文件树
- POST /api/ask      → 流式问答（SSE）
"""

from __future__ import annotations
import hashlib
import logging
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings, setup_logging
from app.schemas import ImportRequest, AskRequest, RepositoryInfo, IndexStatus
from app.ingest import ingest_repository, scan_repository
from app.indexing import MetadataStore, RepoIndex, get_embedding_provider
from app.retrieval import RetrievalPipeline
from app.qa import handle_question

# 初始化日志系统
setup_logging()
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用实例
app = FastAPI(title="OpsWiki", version="0.1.0")

# ── 静态文件与模板配置 ──
FRONTEND_DIR = Path(__file__).parent / "frontend"
# 挂载静态资源目录（CSS、JS）
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
# Jinja2 模板目录
templates = Jinja2Templates(directory=str(FRONTEND_DIR / "templates"))

# ── 全局状态（单进程内共享） ──
_store: MetadataStore | None = None        # SQLite 元数据存储（懒加载单例）
_indexes: dict[str, RepoIndex] = {}        # 仓库 ID → FAISS+BM25 索引
_pipelines: dict[str, RetrievalPipeline] = {}  # 仓库 ID → 检索流水线
_embedder = None                           # Embedding 提供者（懒加载单例）


def get_store() -> MetadataStore:
    """获取全局 MetadataStore 单例（懒加载：首次调用时创建）。"""
    global _store
    if _store is None:
        _store = MetadataStore(settings.data_path / "metadata.db")
    return _store


def get_embedder():
    """获取全局 Embedding 提供者单例。"""
    global _embedder
    if _embedder is None:
        _embedder = get_embedding_provider()
    return _embedder


# ══════════════════════════════════════════════════════════════
#  页面路由
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """渲染主页面（DeepWiki 风格的三栏布局）。"""
    return templates.TemplateResponse(request=request, name="index.html")


# ══════════════════════════════════════════════════════════════
#  API：列出所有仓库
# ══════════════════════════════════════════════════════════════

@app.get("/api/repos")
async def list_repos():
    """列出所有已索引的仓库，供前端选择页展示。"""
    store = get_store()
    return {"repos": store.list_repos()}


# ══════════════════════════════════════════════════════════════
#  API：导入仓库
# ══════════════════════════════════════════════════════════════

@app.post("/api/import")
async def import_repo(req: ImportRequest):
    """
    导入并索引一个本地代码仓库。

    完整流程：
    1. 路径安全校验（存在性、是否为目录）
    2. 生成仓库 ID（基于路径 hash）
    3. 检查是否已索引（幂等处理）
    4. 扫描文件 → AST/Markdown 解析 → 结构化分块
    5. 保存 chunks 和 edges 到 SQLite
    6. 生成向量 → 构建 FAISS + BM25 索引
    7. 初始化检索流水线
    """
    # 将用户输入的路径解析为绝对路径
    path = Path(req.path).resolve()

    # ── 安全校验 ──
    if not path.exists():
        raise HTTPException(400, f"路径不存在: {req.path}")
    if not path.is_dir():
        raise HTTPException(400, f"路径不是目录: {req.path}")

    # 基于路径生成唯一的仓库 ID（12 字符 hex）
    repo_id = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    store = get_store()

    # 幂等性检查：如果已经索引完成，直接返回
    existing = store.get_repo(repo_id)
    if existing and existing["status"] == "ready":
        return {"repo_id": repo_id, "status": "ready", "message": "已有索引，无需重复导入"}

    # 标记状态为"正在索引"
    store.save_repo(repo_id, path.name, str(path), 0, 0, IndexStatus.INDEXING)

    try:
        # 步骤 1：扫描仓库文件
        files = scan_repository(path)
        logger.info(f"扫描 {path}: 发现 {len(files)} 个文件")

        # 步骤 2：解析并分块
        chunks, edges = ingest_repository(path, repo_id)
        store.save_chunks(chunks)
        store.save_edges(repo_id, edges)

        # 步骤 3：构建向量 + BM25 索引
        embedder = get_embedder()
        idx = RepoIndex(repo_id, settings.data_path)
        idx.build(chunks, embedder)
        _indexes[repo_id] = idx

        # 步骤 4：初始化检索流水线
        _pipelines[repo_id] = RetrievalPipeline(idx, store, embedder)

        # 更新仓库状态为"就绪"
        store.save_repo(repo_id, path.name, str(path), len(files), len(chunks), IndexStatus.READY)
        return {
            "repo_id": repo_id,
            "status": "ready",
            "file_count": len(files),
            "chunk_count": len(chunks),
        }

    except Exception as e:
        logger.exception(f"导入失败: {path}")
        store.save_repo(repo_id, path.name, str(path), 0, 0, IndexStatus.FAILED)
        raise HTTPException(500, f"导入失败: {str(e)}")


# ══════════════════════════════════════════════════════════════
#  API：仓库状态查询
# ══════════════════════════════════════════════════════════════

@app.get("/api/repo/{repo_id}")
async def get_repo_info(repo_id: str):
    """获取指定仓库的概览信息（名称、路径、文件数、分块数、状态）。"""
    store = get_store()
    info = store.get_repo(repo_id)
    if not info:
        raise HTTPException(404, "仓库未找到")
    return info


@app.get("/api/repo/{repo_id}/files")
async def get_file_tree(repo_id: str):
    """获取指定仓库的文件列表，用于前端文件导航面板。"""
    store = get_store()
    return {"files": store.get_file_tree(repo_id)}


# ══════════════════════════════════════════════════════════════
#  API：流式问答（SSE）
# ══════════════════════════════════════════════════════════════

@app.post("/api/ask")
async def ask_question(req: AskRequest):
    """
    处理用户提问，以 SSE（Server-Sent Events）流式返回回答。

    SSE 协议：
    - 每条消息格式为 "data: {json}\n\n"
    - 消息类型：citations（引用列表）→ token（逐字输出）→ done（完成信号）

    流程：
    1. 校验仓库存在且索引就绪
    2. 确保检索流水线已初始化（支持服务重启后从磁盘恢复）
    3. 调用 handle_question 进行分类、检索和流式生成
    """
    store = get_store()
    repo_info = store.get_repo(req.repo_id)
    if not repo_info:
        raise HTTPException(404, "仓库未找到")
    if repo_info["status"] != "ready":
        raise HTTPException(400, "仓库索引尚未就绪")

    # 确保检索流水线已就绪（处理服务重启的场景）
    if req.repo_id not in _pipelines:
        embedder = get_embedder()
        chunks = store.get_chunks_by_repo(req.repo_id)
        idx = RepoIndex(req.repo_id, settings.data_path)
        if not idx.load(chunks):
            raise HTTPException(400, "索引文件丢失，请重新导入仓库")
        _indexes[req.repo_id] = idx
        _pipelines[req.repo_id] = RetrievalPipeline(idx, store, embedder)

    pipeline = _pipelines[req.repo_id]

    async def event_stream():
        """SSE 事件流生成器。"""
        async for chunk in handle_question(req.question, pipeline, req.repo_id):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ══════════════════════════════════════════════════════════════
#  应用生命周期
# ══════════════════════════════════════════════════════════════

@app.on_event("shutdown")
async def shutdown():
    """应用关闭时清理资源（关闭 SQLite 连接）。"""
    if _store:
        _store.close()
