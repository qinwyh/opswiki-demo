"""
OpsWiki - 问答模块（QA）
========================
实现两条问答链路：
1. 普通问答链：问题分类 → 检索 → LLM 生成 → 流式输出
2. 复杂问题链：问题分解 → 多轮检索 → 证据聚合 → LLM 生成（含 Mermaid 图）

核心功能：
- get_llm：创建 LangChain ChatOpenAI 实例
- stream_answer：SSE 流式输出回答
- handle_question：统一入口，自动分类并选择合适的问答链路
- decompose_question：将复杂问题拆分为多个子问题

关键修复：使用 langchain_core.messages 替代已废弃的 langchain.schema
"""

from __future__ import annotations
import json
import logging
import re
from typing import AsyncGenerator

from langchain_openai import ChatOpenAI
# ⚠️ 关键修复：新版 langchain 已将消息类移至 langchain_core
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.schemas import (
    AnswerResponse, Citation, SearchHit, QuestionCategory,
)
from app.retrieval import classify_question, RetrievalPipeline

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  LLM 实例化
# ══════════════════════════════════════════════════════════════

def get_llm() -> ChatOpenAI:
    """
    创建并返回一个 LangChain ChatOpenAI 实例。
    通过阿里云百炼的 OpenAI 兼容端点接入 Qwen 系列模型。
    """
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.api_key,
        base_url=settings.llm_base_url,
    )


# ══════════════════════════════════════════════════════════════
#  上下文组装
# ══════════════════════════════════════════════════════════════

def _build_context(hits: list[SearchHit], max_tokens: int = 6000) -> str:
    """
    将检索命中的 chunks 组装为结构化上下文文本，供 LLM 使用。

    每个证据项包含：
    - 序号
    - 文件路径
    - 符号名
    - 行号范围
    - chunk 类型
    - 内容文本（截断到 1500 字符以控制总长度）

    通过 max_tokens 参数控制总上下文长度，避免超出模型窗口。
    """
    parts = []
    total_len = 0
    for i, hit in enumerate(hits):
        c = hit.chunk
        # 构造结构化的证据标题行
        header = f"[Evidence {i+1}] {c.rel_path}"
        if c.symbol_name:
            header += f" :: {c.symbol_name}"
        if c.start_line:
            header += f" (L{c.start_line}-{c.end_line})"
        header += f" [{c.chunk_type}]"

        block = f"{header}\n{c.content[:1500]}\n"

        # 控制总长度（粗略按字符数估算，1 token ≈ 4 字符）
        if total_len + len(block) > max_tokens * 4:
            break
        parts.append(block)
        total_len += len(block)

    return "\n---\n".join(parts)


def _build_citations(hits: list[SearchHit]) -> list[Citation]:
    """
    从检索结果构建引用溯源列表。
    每个 Citation 包含文件路径、行号、代码片段等，供前端展示引用卡片。
    """
    return [
        Citation(
            rel_path=h.chunk.rel_path,
            start_line=h.chunk.start_line,
            end_line=h.chunk.end_line,
            snippet=(h.chunk.raw_code or h.chunk.content)[:500],
            chunk_type=h.chunk.chunk_type,
            symbol_name=h.chunk.symbol_name,
            score=h.score,
        )
        for h in hits
    ]


# ══════════════════════════════════════════════════════════════
#  System Prompts（系统提示词）
# ══════════════════════════════════════════════════════════════

# 普通问答的系统提示词
SYSTEM_PROMPT_SIMPLE = """You are an expert code assistant analyzing a local code repository.
Answer the user's question based ONLY on the provided evidence.
Rules:
- If evidence is insufficient, say so clearly
- Use markdown formatting
- Be concise but thorough
- IMPORTANT: When you use information from an evidence block, you MUST insert a citation marker like [1], [2], etc. corresponding to the evidence number (e.g. [Evidence 1] → [1]). Place the marker right after the relevant sentence or code reference.
- Example: "The `parse_python` function handles AST parsing [3] and extracts metadata from each node [3][5]."
- At the end, suggest 2-3 follow-up questions the user might ask
Output format (respond in the SAME LANGUAGE as the user's question):
1. Your answer in markdown, with citation markers like [1] [2] inline
2. A line "---FOLLOWUPS---" followed by suggested questions, one per line"""

# 复杂问题（含 Mermaid 图生成）的系统提示词
SYSTEM_PROMPT_COMPLEX = """You are an expert code architect analyzing a local code repository.
The user asks a complex question requiring multi-step analysis.
Based ONLY on the provided evidence:
1. Provide a detailed textual explanation
2. If appropriate, generate ONE Mermaid diagram (sequence or flowchart)
3. Every node/edge in the diagram MUST correspond to evidence found in the codebase
4. If evidence is insufficient for some parts, explicitly note uncertainty
Rules:
- Use markdown formatting
- Do NOT fabricate call relationships not supported by evidence
- IMPORTANT: When you use information from an evidence block, you MUST insert a citation marker like [1], [2], etc. corresponding to the evidence number. Place the marker right after the relevant sentence.
- At the end, suggest 2-3 follow-up questions

MERMAID DIAGRAM RULES — STRICTLY follow to avoid parse errors:
- Use ONLY `flowchart TD` (for architecture/call graphs) or `sequenceDiagram` (for request flows)
- NEVER use `graph TD` or `graph LR` — always use `flowchart TD` or `flowchart LR`
- NODE IDs: SHORT alphanumeric+underscore only (e.g., parseFunc, ApiHandler, nodeA). NO spaces, NO Chinese characters, NO hyphens, NO dots in IDs
- NODE LABELS: always wrap display text in double-quoted square brackets: nodeA["label text here"]
- For sequenceDiagram: participant names must be simple alphanumeric. Use `participant A as "Display Name"` pattern
- Arrow labels: wrap in double quotes: `A -->|"label"| B`
- Maximum 10 nodes (flowchart) or 10 steps (sequence)
- NO subgraph blocks unless truly needed
- VALIDATE before outputting: every node ID used in arrows must be declared, all quotes must be closed
- Wrap Mermaid code in ```mermaid ... ``` block
- DO NOT output anything after the closing ``` of the mermaid block

Output format (respond in the SAME LANGUAGE as the user's question):
1. Textual explanation with inline citation markers [1] [2] etc.
2. Mermaid diagram (if applicable)
3. A line "---FOLLOWUPS---" followed by suggested questions, one per line"""


# ══════════════════════════════════════════════════════════════
#  响应解析
# ══════════════════════════════════════════════════════════════

def _parse_response(text: str) -> tuple[str, str, list[str]]:
    """
    解析 LLM 的原始文本输出，分离出：
    1. 主回答文本（answer_text）
    2. Mermaid 图代码（如果有）
    3. 建议的后续问题列表

    解析规则：
    - 以 "---FOLLOWUPS---" 为分隔符拆分主文本和后续问题
    - 用正则提取 ```mermaid ... ``` 代码块
    """
    followups: list[str] = []
    main_text = text

    # 分离后续问题
    if "---FOLLOWUPS---" in text:
        parts = text.split("---FOLLOWUPS---")
        main_text = parts[0].strip()
        followup_text = parts[1].strip() if len(parts) > 1 else ""
        followups = [
            line.strip().lstrip("- ").lstrip("0123456789.").strip()
            for line in followup_text.split("\n")
            if line.strip()
        ]

    # 提取 Mermaid 图代码
    mermaid = ""
    mermaid_match = re.search(r'```mermaid\s*\n(.*?)```', main_text, re.DOTALL)
    if mermaid_match:
        mermaid = mermaid_match.group(1).strip()

    return main_text, mermaid, followups[:5]


# ══════════════════════════════════════════════════════════════
#  普通问答链
# ══════════════════════════════════════════════════════════════

async def answer_simple(question: str, hits: list[SearchHit]) -> AnswerResponse:
    """
    普通问答：直接基于检索结果生成回答。
    适用于代码定位、函数说明、配置查找等简单问题。
    """
    llm = get_llm()
    context = _build_context(hits)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT_SIMPLE),
        HumanMessage(content=f"Evidence:\n{context}\n\nQuestion: {question}"),
    ]
    resp = await llm.ainvoke(messages)
    answer_text, mermaid, followups = _parse_response(resp.content)
    return AnswerResponse(
        answer_text=answer_text,
        citations=_build_citations(hits),
        optional_mermaid=mermaid,
        suggested_followups=followups,
        question_category=QuestionCategory.CODE_LOCATE,
    )


# ══════════════════════════════════════════════════════════════
#  复杂问题链（含子任务分解）
# ══════════════════════════════════════════════════════════════

# 问题分解的提示词：让 LLM 将复杂问题拆成 3-5 个可独立检索的子问题
DECOMPOSE_PROMPT = """Given this complex question about a code repository, break it down into 3-5 sub-questions that can each be answered by searching the codebase. Return ONLY a JSON array of strings.
Question: {question}"""


async def decompose_question(question: str) -> list[str]:
    """
    将复杂问题分解为多个子问题。

    流程：
    1. 调用 LLM 生成 JSON 格式的子问题列表
    2. 尝试解析 JSON 数组
    3. 如果 JSON 解析失败，按行分割作为 fallback
    """
    llm = get_llm()
    resp = await llm.ainvoke([HumanMessage(content=DECOMPOSE_PROMPT.format(question=question))])
    try:
        # 尝试从回复中提取 JSON 数组
        json_match = re.search(r'\[.*\]', resp.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    # Fallback：按行分割
    return [
        line.strip().lstrip("- 0123456789.")
        for line in resp.content.split("\n")
        if line.strip()
    ][:5]


async def answer_complex(
    question: str,
    pipeline: RetrievalPipeline,
    repo_id: str,
) -> AnswerResponse:
    """
    复杂问答链：分解问题 → 多轮检索 → 证据聚合 → 生成回答（含 Mermaid 图）。

    适用于：
    - 生成时序图 / 架构图
    - 解释跨文件调用流程
    - 分析某功能从入口到核心逻辑的完整链路
    """
    # 步骤 1：将复杂问题分解为子任务
    sub_questions = await decompose_question(question)
    logger.info(f"问题已分解为 {len(sub_questions)} 个子问题: {sub_questions}")

    # 步骤 2：对每个子问题独立检索，聚合去重
    all_hits: list[SearchHit] = []
    seen_ids: set[str] = set()
    for sq in sub_questions:
        hits = pipeline.retrieve(repo_id, sq, top_k=5)
        for h in hits:
            if h.chunk.chunk_id not in seen_ids:
                seen_ids.add(h.chunk.chunk_id)
                all_hits.append(h)

    # 同时也检索原始问题（可能捕获子问题遗漏的内容）
    orig_hits = pipeline.retrieve(repo_id, question, top_k=8)
    for h in orig_hits:
        if h.chunk.chunk_id not in seen_ids:
            seen_ids.add(h.chunk.chunk_id)
            all_hits.append(h)

    # 步骤 3：调用 LLM 生成含 Mermaid 图的复杂回答
    llm = get_llm()
    context = _build_context(all_hits[:15])
    messages = [
        SystemMessage(content=SYSTEM_PROMPT_COMPLEX),
        HumanMessage(content=(
            f"Evidence:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Sub-questions explored: {json.dumps(sub_questions)}"
        )),
    ]
    resp = await llm.ainvoke(messages)
    answer_text, mermaid, followups = _parse_response(resp.content)

    return AnswerResponse(
        answer_text=answer_text,
        citations=_build_citations(all_hits[:12]),
        optional_mermaid=mermaid,
        suggested_followups=followups,
        question_category=QuestionCategory.COMPLEX_REASONING,
    )


# ══════════════════════════════════════════════════════════════
#  流式输出
# ══════════════════════════════════════════════════════════════

async def stream_answer(
    question: str,
    hits: list[SearchHit],
    is_complex: bool = False,
) -> AsyncGenerator[str, None]:
    """
    以 SSE（Server-Sent Events）格式流式输出 LLM 回答。

    输出协议（每行一个 JSON 消息）：
    1. {"type": "citations", "data": [...]}  —— 首先发送引用列表
    2. {"type": "token", "data": "..."}      —— 逐 token 流式发送回答
    3. {"type": "done", "mermaid": "...", "followups": [...]}  —— 完成信号
    """
    llm = get_llm()
    context = _build_context(hits)
    system = SYSTEM_PROMPT_COMPLEX if is_complex else SYSTEM_PROMPT_SIMPLE
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Evidence:\n{context}\n\nQuestion: {question}"),
    ]

    # 第一步：立即发送引用列表（前端可以先渲染引用面板）
    citations = _build_citations(hits)
    yield json.dumps({"type": "citations", "data": [c.model_dump() for c in citations]}) + "\n"

    # 第二步：流式发送 LLM 生成的 token
    full_text = ""
    async for chunk in llm.astream(messages):
        token = chunk.content
        if token:
            full_text += token
            yield json.dumps({"type": "token", "data": token}) + "\n"

    # 第三步：解析完整回答，发送完成信号（含 Mermaid 图和后续问题）
    _, mermaid, followups = _parse_response(full_text)
    yield json.dumps({"type": "done", "mermaid": mermaid, "followups": followups}) + "\n"


# ══════════════════════════════════════════════════════════════
#  统一入口
# ══════════════════════════════════════════════════════════════

async def handle_question(
    question: str,
    pipeline: RetrievalPipeline,
    repo_id: str,
) -> AsyncGenerator[str, None]:
    """
    问答的统一入口：自动分类问题并选择合适的处理链路。

    普通问题：直接检索 + 流式回答
    复杂问题：分解子任务 → 多轮检索 → 聚合证据 → 流式回答（含 Mermaid 图）
    """
    category = classify_question(question)
    logger.info(f"处理问题 [类型={category}]: {question[:80]}")

    if category == QuestionCategory.COMPLEX_REASONING:
        # ── 复杂问题链 ──
        # 步骤 1：分解问题
        sub_questions = await decompose_question(question)

        # 步骤 2：多轮检索并聚合
        all_hits: list[SearchHit] = []
        seen: set[str] = set()
        for sq in sub_questions:
            for h in pipeline.retrieve(repo_id, sq, top_k=5):
                if h.chunk.chunk_id not in seen:
                    seen.add(h.chunk.chunk_id)
                    all_hits.append(h)
        # 同时检索原始问题
        for h in pipeline.retrieve(repo_id, question, top_k=8):
            if h.chunk.chunk_id not in seen:
                seen.add(h.chunk.chunk_id)
                all_hits.append(h)

        # 步骤 3：流式输出
        async for chunk in stream_answer(question, all_hits[:15], is_complex=True):
            yield chunk
    else:
        # ── 普通问题链 ──
        hits = pipeline.retrieve(repo_id, question)
        async for chunk in stream_answer(question, hits, is_complex=False):
            yield chunk
