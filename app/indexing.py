"""
OpsWiki - 索引模块（Indexing）
==============================
负责向量索引、BM25 稀疏索引、SQLite 元数据存储。

核心组件：
- EmbeddingProvider：向量嵌入接口（Protocol），方便切换不同 embedding 后端
- DashscopeEmbedding：基于阿里云百炼 OpenAI 兼容 API 的默认实现
- MetadataStore：SQLite 元数据存储（chunks 表、edges 表、repos 表）
- RepoIndex：单仓库的 FAISS 向量索引 + BM25 稀疏索引
"""

from __future__ import annotations
import json
import logging
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from app.config import settings
from app.schemas import Chunk, CodeEdge

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  Embedding 抽象接口与实现
# ══════════════════════════════════════════════════════════════

class EmbeddingProvider(Protocol):
    """
    向量嵌入提供者的抽象接口（Python Protocol）。
    任何实现了 embed_texts 和 dimension 的类都自动满足此接口。
    便于后续替换为本地模型（如 jina-embeddings-v2-base-code）。
    """
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


class DashscopeEmbedding:
    """
    通过阿里云百炼（Dashscope）的 OpenAI 兼容 API 生成文本向量。
    默认使用 text-embedding-v3 模型，维度 1024。
    """

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.embedding_base_url,
        )
        self.model = settings.embedding_model
        self._dim = settings.embedding_dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        批量生成文本向量。

        处理细节：
        - 每批最多 20 条文本（API 限制）
        - 超长文本截断到 8000 字符（避免超出 token 限制）
        - 返回 shape = (len(texts), dimension) 的 float32 数组
        """
        all_embeddings = []
        batch_size = 10  # Dashscope API 限制每批最多 10 条
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # 截断过长文本，避免超出 API token 限制
            batch = [t[:8000] for t in batch]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            for item in resp.data:
                all_embeddings.append(item.embedding)
        return np.array(all_embeddings, dtype=np.float32)


def get_embedding_provider() -> EmbeddingProvider:
    """工厂函数：根据配置返回对应的 embedding 提供者实例。"""
    if settings.embedding_provider == "dashscope":
        return DashscopeEmbedding()
    raise ValueError(f"未知的 embedding 提供者: {settings.embedding_provider}")


# ══════════════════════════════════════════════════════════════
#  SQLite 元数据存储
# ══════════════════════════════════════════════════════════════

class MetadataStore:
    """
    基于 SQLite 的元数据持久化存储。

    表结构：
    - chunks：存储所有代码/文档分块及其完整元数据
    - edges：存储代码关系边（contains、imports、calls）
    - repos：存储仓库概览信息与索引状态

    索引设计：
    - 在 repo_id、rel_path、symbol_name、source、target 上建索引
    - 支持精确匹配检索（symbol_name LIKE 查询）
    - 支持 1-hop 图扩展（通过 edges 表查邻居）
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        # check_same_thread=False 允许多线程访问（FastAPI 是异步多协程的）
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # WAL 模式提升并发读写性能
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self):
        """初始化数据库表结构（幂等操作，重复调用安全）。"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                repo_id TEXT,
                rel_path TEXT,
                language TEXT,
                chunk_type TEXT,
                symbol_name TEXT,
                qualified_name TEXT,
                parent_class TEXT,
                signature TEXT,
                docstring TEXT,
                start_line INTEGER,
                end_line INTEGER,
                imports TEXT,
                outgoing_calls TEXT,
                raw_code TEXT,
                content TEXT,
                heading_path TEXT
            );
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id TEXT,
                source TEXT,
                target TEXT,
                relation TEXT
            );
            CREATE TABLE IF NOT EXISTS repos (
                repo_id TEXT PRIMARY KEY,
                name TEXT,
                path TEXT,
                file_count INTEGER,
                chunk_count INTEGER,
                status TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(rel_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_name);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
        """)
        self.conn.commit()

    def save_chunks(self, chunks: list[Chunk]):
        """批量保存 chunks 到数据库（INSERT OR REPLACE 支持幂等更新）。"""
        self.conn.executemany(
            """INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [(c.chunk_id, c.repo_id, c.rel_path, c.language, c.chunk_type.value,
              c.symbol_name, c.qualified_name, c.parent_class, c.signature,
              c.docstring, c.start_line, c.end_line,
              json.dumps(c.imports), json.dumps(c.outgoing_calls),
              c.raw_code, c.content, c.heading_path) for c in chunks]
        )
        self.conn.commit()

    def save_edges(self, repo_id: str, edges: list[CodeEdge]):
        """批量保存代码关系边到数据库。"""
        self.conn.executemany(
            "INSERT INTO edges (repo_id, source, target, relation) VALUES (?,?,?,?)",
            [(repo_id, e.source, e.target, e.relation) for e in edges]
        )
        self.conn.commit()

    def save_repo(self, repo_id: str, name: str, path: str, file_count: int, chunk_count: int, status: str):
        """保存/更新仓库信息。"""
        self.conn.execute(
            "INSERT OR REPLACE INTO repos VALUES (?,?,?,?,?,?)",
            (repo_id, name, path, file_count, chunk_count, status)
        )
        self.conn.commit()

    def get_repo(self, repo_id: str) -> dict | None:
        """根据 repo_id 获取仓库信息。"""
        row = self.conn.execute("SELECT * FROM repos WHERE repo_id=?", (repo_id,)).fetchone()
        if not row:
            return None
        return dict(zip(["repo_id", "name", "path", "file_count", "chunk_count", "status"], row))

    def list_repos(self) -> list[dict]:
        """列出所有仓库（按导入时间倒序）。"""
        rows = self.conn.execute("SELECT * FROM repos ORDER BY rowid DESC").fetchall()
        cols = ["repo_id", "name", "path", "file_count", "chunk_count", "status"]
        return [dict(zip(cols, r)) for r in rows]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """根据 chunk_id 获取单个 chunk。"""
        row = self.conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        if not row:
            return None
        return self._row_to_chunk(row)

    def get_chunks_by_repo(self, repo_id: str) -> list[Chunk]:
        """获取某仓库的所有 chunks（用于重建索引时加载数据）。"""
        rows = self.conn.execute("SELECT * FROM chunks WHERE repo_id=?", (repo_id,)).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def search_exact(self, repo_id: str, query: str, limit: int = 10) -> list[Chunk]:
        """
        精确检索：在 symbol_name、rel_path、qualified_name 上做 LIKE 模糊匹配。
        这是三路召回中的 "exact retrieval" 通道。
        """
        q = f"%{query}%"
        rows = self.conn.execute(
            """SELECT * FROM chunks WHERE repo_id=? AND
               (symbol_name LIKE ? OR rel_path LIKE ? OR qualified_name LIKE ?)
               LIMIT ?""",
            (repo_id, q, q, q, limit)
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def get_neighbors(self, repo_id: str, symbol: str, limit: int = 5) -> list[Chunk]:
        """
        轻量图扩展：通过 edges 表做 1-hop 邻域查找。

        流程：
        1. 在 edges 表中找到 source 或 target 包含 symbol 的边
        2. 收集所有相关的节点名称
        3. 在 chunks 表中查找这些节点对应的 chunk

        用例示例：
        - 命中函数 → 补充其所在文件摘要
        - 命中文件摘要 → 补充其中主要函数
        - 命中函数 → 补充其调用的其他函数
        """
        # 步骤 1：查找相关的边
        edge_rows = self.conn.execute(
            """SELECT source, target, relation FROM edges
               WHERE repo_id=? AND (source LIKE ? OR target LIKE ?) LIMIT 20""",
            (repo_id, f"%{symbol}%", f"%{symbol}%")
        ).fetchall()

        # 步骤 2：收集相关节点名
        related_names = set()
        for src, tgt, rel in edge_rows:
            related_names.add(src)
            related_names.add(tgt)
        related_names.discard(symbol)  # 排除自身

        # 步骤 3：查找对应的 chunks
        results = []
        for name in list(related_names)[:limit]:
            rows = self.conn.execute(
                "SELECT * FROM chunks WHERE repo_id=? AND (symbol_name=? OR rel_path=? OR qualified_name=?) LIMIT 2",
                (repo_id, name, name, name)
            ).fetchall()
            results.extend([self._row_to_chunk(r) for r in rows])
        return results[:limit]

    def _row_to_chunk(self, row) -> Chunk:
        """将 SQLite 查询结果行转换为 Chunk 对象。"""
        return Chunk(
            chunk_id=row[0], repo_id=row[1], rel_path=row[2], language=row[3],
            chunk_type=row[4], symbol_name=row[5], qualified_name=row[6],
            parent_class=row[7], signature=row[8], docstring=row[9],
            start_line=row[10], end_line=row[11],
            imports=json.loads(row[12]) if row[12] else [],
            outgoing_calls=json.loads(row[13]) if row[13] else [],
            raw_code=row[14], content=row[15], heading_path=row[16] or "",
        )

    def get_file_tree(self, repo_id: str) -> list[str]:
        """获取仓库的文件列表（去重、排序），用于前端文件导航展示。"""
        rows = self.conn.execute(
            "SELECT DISTINCT rel_path FROM chunks WHERE repo_id=? ORDER BY rel_path", (repo_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def close(self):
        """关闭数据库连接。"""
        self.conn.close()


# ══════════════════════════════════════════════════════════════
#  向量 + BM25 混合索引
# ══════════════════════════════════════════════════════════════

class RepoIndex:
    """
    单仓库的混合检索索引，包含：
    - FAISS 向量索引（IndexFlatIP，L2 归一化后等价于余弦相似度）
    - BM25 稀疏索引（基于 rank-bm25 库）

    持久化策略：
    - FAISS 索引保存为 faiss.index 文件
    - chunk ID 映射保存为 chunk_ids.json
    - BM25 无需持久化，从 chunks 内容重建（速度很快）
    """

    def __init__(self, repo_id: str, data_dir: Path):
        self.repo_id = repo_id
        self.dir = data_dir / repo_id  # 每个仓库一个子目录
        self.dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []  # 位置 i 对应 FAISS 中第 i 个向量的 chunk_id

    def build(self, chunks: list[Chunk], embedder: EmbeddingProvider):
        """
        构建 FAISS + BM25 双索引。

        流程：
        1. 收集所有 chunk 的 content 文本
        2. 调用 embedding API 生成向量
        3. L2 归一化后存入 FAISS（内积检索 = 余弦相似度）
        4. 对文本分词后构建 BM25 索引
        5. 持久化到磁盘
        """
        if not chunks:
            return
        self.chunk_ids = [c.chunk_id for c in chunks]
        texts = [c.content for c in chunks]

        # ── FAISS 向量索引 ──
        logger.info(f"正在为 {len(texts)} 个分块生成向量...")
        vectors = embedder.embed_texts(texts)
        # L2 归一化：归一化后的内积 == 余弦相似度
        faiss.normalize_L2(vectors)
        self.faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        self.faiss_index.add(vectors)

        # ── BM25 稀疏索引 ──
        # 简单空格分词（对代码和英文效果尚可，后续可替换为更好的分词器）
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

        # 持久化索引文件
        self._save()
        logger.info(f"索引构建完成 [{self.repo_id}]: {len(chunks)} 个分块")

    def search_dense(self, query_vec: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """
        向量检索：使用 FAISS 进行 top-K 近邻搜索。

        参数：
            query_vec: 查询向量，shape = (1, dimension)
            top_k: 返回最相似的 K 个结果

        返回：[(chunk_id, 相似度分数), ...]
        """
        if self.faiss_index is None:
            return []
        faiss.normalize_L2(query_vec)  # 查询向量也需要归一化
        scores, indices = self.faiss_index.search(query_vec, min(top_k, self.faiss_index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            results.append((self.chunk_ids[idx], float(score)))
        return results

    def search_sparse(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        BM25 稀疏检索：基于词频的传统信息检索。
        对精确关键词匹配特别有效（如函数名、类名）。

        返回：[(chunk_id, BM25分数), ...]，按分数降序
        """
        if self.bm25 is None:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        # 按分数降序取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0 and idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(scores[idx])))
        return results

    def _save(self):
        """将 FAISS 索引和 chunk ID 映射持久化到磁盘。"""
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(self.dir / "faiss.index"))
        with open(self.dir / "chunk_ids.json", "w") as f:
            json.dump(self.chunk_ids, f)

    def load(self, chunks: list[Chunk]) -> bool:
        """
        从磁盘加载已有的索引。

        参数：
            chunks: 对应仓库的所有 chunks（用于重建 BM25）

        返回：True 表示加载成功，False 表示索引文件不存在
        """
        faiss_path = self.dir / "faiss.index"
        ids_path = self.dir / "chunk_ids.json"
        if not faiss_path.exists() or not ids_path.exists():
            return False

        # 加载 FAISS 索引
        self.faiss_index = faiss.read_index(str(faiss_path))

        # 加载 chunk ID 映射
        with open(ids_path) as f:
            self.chunk_ids = json.load(f)

        # 重建 BM25 索引（BM25 不支持序列化，但重建很快）
        id_to_chunk = {c.chunk_id: c for c in chunks}
        texts = [id_to_chunk[cid].content if cid in id_to_chunk else "" for cid in self.chunk_ids]
        tokenized = [t.lower().split() for t in texts]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)
        return True
