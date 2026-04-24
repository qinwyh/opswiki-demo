"""
OpsWiki - 检索模块（Retrieval）
================================
实现完整的 RAG 检索链路：
查询理解 → 查询翻译 → 多路召回(Dense+Sparse+Exact) → RRF融合 → 重排 → 图扩展

核心组件：
- classify_question：问题分类（代码定位 / 文档架构 / 复杂推理）
- extract_keywords：从问题中提取符号名、文件路径、代码关键词
- translate_query：将用户问题翻译为多种内部查询变体
- rrf_fusion：Reciprocal Rank Fusion 多路结果融合
- HeuristicReranker：基于启发式特征的二阶段重排（接口预留 Cross-Encoder）
- RetrievalPipeline：完整的检索流水线，串联以上所有步骤
"""

from __future__ import annotations
import logging
import re
from typing import Protocol

import numpy as np

from app.config import settings
from app.schemas import Chunk, SearchHit, QuestionCategory
from app.indexing import RepoIndex, MetadataStore, EmbeddingProvider

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  重排器抽象接口与实现
# ══════════════════════════════════════════════════════════════

class Reranker(Protocol):
    """
    重排器抽象接口（Protocol）。
    后续可替换为真正的 Cross-Encoder 模型（如 ms-marco-MiniLM）。
    """
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]: ...


class HeuristicReranker:
    """
    基于启发式特征的 Fallback 重排器。

    打分特征：
    1. 查询词与 chunk 内容的 token 重叠度
    2. 符号名是否出现在查询中（强信号）
    3. 文件路径是否包含查询关键词
    4. chunk 类型偏好（函数/类 > 文件摘要）
    5. 是否有 docstring（有文档 = 质量更高）
    """

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        scored = []

        for chunk in chunks:
            score = 0.0
            content_lower = chunk.content.lower()

            # 特征 1：token 重叠度
            chunk_tokens = set(content_lower.split())
            overlap = len(query_tokens & chunk_tokens)
            score += overlap * 0.1

            # 特征 2：符号名直接匹配（强信号，+2 分）
            if chunk.symbol_name and chunk.symbol_name.lower() in query_lower:
                score += 2.0

            # 特征 3：路径关键词匹配
            for token in query_tokens:
                if token in chunk.rel_path.lower():
                    score += 1.0

            # 特征 4：偏好代码结构（函数/方法/类）而非文件摘要
            if chunk.chunk_type in ("function", "method", "class"):
                score += 0.5

            # 特征 5：有 docstring 的 chunk 通常更有价值
            if chunk.docstring:
                score += 0.3

            scored.append((chunk, score))

        # 按分数降序排列，取 top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class LLMReranker:
    """
    基于 LLM 的重排器（预留接口）。
    当前 MVP 阶段直接委托给启发式重排器，避免增加 API 调用成本。
    后续可实现真正的 LLM-based reranking（让模型给每个候选打相关性分）。
    """

    def __init__(self, llm):
        self.llm = llm

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        # MVP 阶段直接使用启发式重排
        return HeuristicReranker().rerank(query, chunks, top_k)


# ══════════════════════════════════════════════════════════════
#  查询理解
# ══════════════════════════════════════════════════════════════

def classify_question(question: str) -> QuestionCategory:
    """
    对用户问题做轻量分类，决定走哪条问答链路。

    分类逻辑：
    1. 包含"时序图"、"架构图"、"mermaid"等关键词 → 复杂推理
    2. 包含"文档"、"readme"、"说明"等关键词 → 文档/架构类
    3. 其他 → 代码定位类（默认）
    """
    q = question.lower()

    # 复杂推理类信号词
    complex_signals = [
        "时序图", "架构图", "sequence diagram", "architecture", "mermaid",
        "调用流程", "调用链", "全流程", "整体", "系统", "关系图",
        "how does .* work end to end", "explain the flow",
    ]
    for sig in complex_signals:
        if re.search(sig, q):
            return QuestionCategory.COMPLEX_REASONING

    # 文档/架构类信号词
    doc_signals = ["文档", "readme", "说明", "架构", "设计", "部署", "documentation"]
    for sig in doc_signals:
        if sig in q:
            return QuestionCategory.DOC_ARCH

    # 默认：代码定位类
    return QuestionCategory.CODE_LOCATE


def extract_keywords(question: str) -> dict:
    """
    从用户问题中提取可能的代码关键词，用于精确检索和查询增强。

    提取内容：
    - symbols：反引号包裹的内容（如 `parse_python`）
    - code_words：CamelCase 或 snake_case 风格的词
    - paths：看起来像文件路径的内容（如 app/ingest.py）
    - raw_tokens：原始分词结果
    """
    # 提取反引号中的符号名
    symbols = re.findall(r'`([^`]+)`', question)
    # 识别 CamelCase 或 snake_case 风格的代码词
    code_words = re.findall(r'\b([A-Z][a-zA-Z]+|[a-z]+_[a-z_]+)\b', question)
    # 识别文件路径模式
    paths = re.findall(r'[\w/\\]+\.\w+', question)

    return {
        "symbols": symbols,
        "code_words": list(set(code_words))[:10],
        "paths": paths,
        "raw_tokens": question.lower().split(),
    }


def translate_query(question: str, keywords: dict) -> list[dict]:
    """
    查询翻译：将用户问题转换为多种内部查询变体。

    目的：不同类型的查询适合不同的检索通道：
    - natural：原始自然语言 → 适合向量检索（语义匹配）
    - keyword：代码关键词 → 适合 BM25 检索（精确词匹配）
    - path：文件路径 → 适合精确检索
    - symbol：符号名 → 适合精确检索
    """
    queries = [
        {"type": "natural", "text": question},
        {"type": "keyword", "text": " ".join(
            keywords.get("code_words", []) + keywords.get("symbols", [])
        )},
    ]
    if keywords.get("paths"):
        queries.append({"type": "path", "text": " ".join(keywords["paths"])})
    if keywords.get("symbols"):
        queries.append({"type": "symbol", "text": " ".join(keywords["symbols"])})
    return queries


# ══════════════════════════════════════════════════════════════
#  RRF 融合
# ══════════════════════════════════════════════════════════════

def rrf_fusion(result_lists: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion（RRF）—— 将多路召回结果融合为统一排序。

    原理：对每路结果中的每个文档，按其排名计算 1/(k + rank) 分数，
    然后将同一文档在不同路中的分数求和。

    参数：
        result_lists: 多路召回结果，每路是 [(chunk_id, score)] 列表
        k: 平滑常数（默认 60，来自原始 RRF 论文）

    返回：融合后的 [(chunk_id, rrf_score)] 列表，按分数降序
    """
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, (chunk_id, _) in enumerate(results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items


# ══════════════════════════════════════════════════════════════
#  完整检索流水线
# ══════════════════════════════════════════════════════════════

class RetrievalPipeline:
    """
    完整的 RAG 检索流水线，串联所有检索步骤。

    流程：
    1. 查询理解：分类问题、提取关键词
    2. 查询翻译：生成多种查询变体
    3. 多路召回：Dense(FAISS) + Sparse(BM25) + Exact(SQLite) 并行检索
    4. RRF 融合：将三路结果合并为统一排序
    5. 获取完整 chunk 数据
    6. 二阶段重排：使用启发式/模型重排器精排
    7. 轻量图扩展：对高分 chunk 做 1-hop 邻域补充
    """

    def __init__(
        self,
        index: RepoIndex,
        store: MetadataStore,
        embedder: EmbeddingProvider,
        reranker: Reranker | None = None,
    ):
        self.index = index
        self.store = store
        self.embedder = embedder
        self.reranker = reranker or HeuristicReranker()

    def retrieve(self, repo_id: str, question: str, top_k: int | None = None) -> list[SearchHit]:
        """
        执行完整的检索流程。

        参数：
            repo_id: 仓库 ID
            question: 用户问题
            top_k: 最终返回的结果数量

        返回：SearchHit 列表，按相关性降序
        """
        top_k = top_k or settings.rerank_top_k
        retrieval_k = settings.retrieval_top_k

        # ── 步骤 1：查询理解 ──
        category = classify_question(question)
        keywords = extract_keywords(question)
        queries = translate_query(question, keywords)
        logger.info(f"查询分类={category}, 关键词={keywords}")

        # ── 步骤 2：多路并行召回 ──
        result_lists: list[list[tuple[str, float]]] = []

        # 通道 A：Dense 向量检索（语义匹配）
        q_vec = self.embedder.embed_texts([question])
        dense_results = self.index.search_dense(q_vec, retrieval_k)
        result_lists.append(dense_results)

        # 通道 B：Sparse BM25 检索（关键词匹配），使用多个查询变体
        for q in queries:
            sparse_results = self.index.search_sparse(q["text"], retrieval_k)
            if sparse_results:
                result_lists.append(sparse_results)

        # 通道 C：Exact 精确检索（符号名/路径匹配）
        exact_chunks = []
        for sym in keywords.get("symbols", []) + keywords.get("code_words", [])[:5]:
            exact_chunks.extend(self.store.search_exact(repo_id, sym, limit=5))
        if exact_chunks:
            result_lists.append([(c.chunk_id, 1.0) for c in exact_chunks])

        # ── 步骤 3：RRF 融合 ──
        fused = rrf_fusion(result_lists)

        # ── 步骤 4：获取完整 chunk 数据（去重） ──
        candidates: list[Chunk] = []
        seen = set()
        for chunk_id, _ in fused[:retrieval_k * 2]:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk = self.store.get_chunk(chunk_id)
            if chunk:
                candidates.append(chunk)

        # ── 步骤 5：二阶段重排 ──
        reranked = self.reranker.rerank(question, candidates, top_k)

        # ── 步骤 6：轻量图扩展 ──
        # 对重排后 top-3 的 chunk，查找其 1-hop 邻居并补充到结果中
        expanded = list(reranked)
        if settings.graph_expand_hops > 0 and reranked:
            existing_ids = {c.chunk_id for c, _ in reranked}
            for chunk, score in reranked[:3]:  # 只对 top-3 做扩展，控制 token 预算
                neighbors = self.store.get_neighbors(
                    repo_id,
                    chunk.symbol_name or chunk.rel_path,
                    limit=3
                )
                for n in neighbors:
                    if n.chunk_id not in existing_ids:
                        existing_ids.add(n.chunk_id)
                        # 邻居的分数衰减为原始分数的 50%
                        expanded.append((n, score * 0.5))

        # ── 步骤 7：构建最终 SearchHit 列表 ──
        hits: list[SearchHit] = []
        for chunk, score in expanded[:top_k + 4]:  # 略多于 top_k，保留多样性
            hits.append(SearchHit(chunk=chunk, score=score))

        return hits
