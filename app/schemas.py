"""
OpsWiki - 数据模型（Schemas）
============================
定义整个系统中流转的核心数据结构，包括：
- 代码分块（Chunk）
- 代码关系边（CodeEdge）
- 引用溯源（Citation）
- 问答响应（AnswerResponse）
- 仓库信息（RepositoryInfo）
- API 请求/响应模型

所有模型基于 Pydantic v2，确保类型安全与自动序列化。
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


# ══════════════════════════════════════════════════════════════
#  枚举类型
# ══════════════════════════════════════════════════════════════

class ChunkType(str, Enum):
    """代码分块的类型枚举，用于区分不同粒度的代码片段。"""
    FILE_SUMMARY = "file_summary"   # 文件级摘要（包含 import、顶层定义等概览）
    CLASS = "class"                 # 类定义
    FUNCTION = "function"           # 顶层函数定义
    METHOD = "method"               # 类中的方法定义
    MODULE = "module"               # 整个模块（AST 解析失败时的 fallback）
    MD_SECTION = "md_section"       # Markdown 按标题分块后的单个章节


class QuestionCategory(str, Enum):
    """用户问题分类枚举，决定走哪条问答链路。"""
    CODE_LOCATE = "code_locate"           # 代码定位类（某函数在哪、做什么）
    DOC_ARCH = "doc_arch"                 # 文档/架构类（README 说了什么）
    COMPLEX_REASONING = "complex_reasoning"  # 复杂推理类（生成时序图、架构图）


class IndexStatus(str, Enum):
    """仓库索引状态枚举。"""
    IDLE = "idle"           # 空闲（未导入）
    INDEXING = "indexing"   # 正在建立索引
    READY = "ready"         # 索引就绪，可提问
    FAILED = "failed"       # 索引失败


# ══════════════════════════════════════════════════════════════
#  核心数据结构
# ══════════════════════════════════════════════════════════════

class Chunk(BaseModel):
    """
    代码/文档分块 —— 系统中最核心的数据单元。

    每个 Chunk 对应仓库中的一个语义单元：
    - Python: 一个函数、一个类、一个方法、一个文件摘要
    - Markdown: 一个标题下的章节

    所有字段都携带丰富的元数据，用于检索排序和引用溯源。
    """
    chunk_id: str = ""              # 全局唯一 ID（基于内容 hash）
    repo_id: str = ""               # 所属仓库 ID
    rel_path: str = ""              # 相对于仓库根目录的文件路径
    language: str = ""              # 编程语言（python / markdown）
    chunk_type: ChunkType = ChunkType.MODULE  # 分块类型
    symbol_name: str = ""           # 符号名（函数名、类名、章节标题）
    qualified_name: str = ""        # 完全限定名（如 ClassName.method_name）
    parent_class: str = ""          # 所属类名（仅 method 类型有值）
    signature: str = ""             # 函数/方法签名（如 def foo(x: int) -> str）
    docstring: str = ""             # 文档字符串
    start_line: int = 0             # 在源文件中的起始行号
    end_line: int = 0               # 在源文件中的结束行号
    imports: list[str] = Field(default_factory=list)          # 该文件/块的 import 列表
    outgoing_calls: list[str] = Field(default_factory=list)   # 该函数/方法中调用的其他符号
    raw_code: str = ""              # 原始代码文本（用于引用展示）
    content: str = ""               # 用于 embedding 和检索的文本内容
    heading_path: str = ""          # Markdown 专用：标题层级路径（如 "安装 > 配置 > 环境变量"）


class CodeEdge(BaseModel):
    """
    代码关系边 —— 记录代码实体之间的轻量关系。
    用于检索后的 1-hop 图扩展（例如：命中函数 → 补充其所在文件摘要）。
    """
    source: str = ""    # 源节点（文件路径或符号名）
    target: str = ""    # 目标节点
    relation: str = ""  # 关系类型：contains（包含）、imports（导入）、calls（调用）


class Citation(BaseModel):
    """
    引用溯源 —— 回答中每条证据的元信息。
    前端据此展示引用卡片，用户可查看对应的代码片段与行号。
    """
    rel_path: str               # 文件相对路径
    start_line: int = 0         # 起始行号
    end_line: int = 0           # 结束行号
    snippet: str = ""           # 代码/文本片段预览
    chunk_type: str = ""        # 分块类型
    symbol_name: str = ""       # 符号名
    score: float = 0.0          # 检索得分（用于排序和调试）


class AnswerResponse(BaseModel):
    """
    问答响应 —— LLM 生成的完整回答结构。
    包含回答文本、引用列表、可选 Mermaid 图和建议的后续问题。
    """
    answer_text: str = ""                                           # Markdown 格式的回答正文
    citations: list[Citation] = Field(default_factory=list)         # 引用证据列表
    optional_mermaid: str = ""                                      # 可选的 Mermaid 图代码
    suggested_followups: list[str] = Field(default_factory=list)    # 建议的后续问题
    question_category: str = ""                                     # 问题分类结果


class RepositoryInfo(BaseModel):
    """仓库概览信息。"""
    repo_id: str = ""
    name: str = ""
    path: str = ""
    file_count: int = 0
    chunk_count: int = 0
    status: IndexStatus = IndexStatus.IDLE


# ══════════════════════════════════════════════════════════════
#  API 请求模型
# ══════════════════════════════════════════════════════════════

class ImportRequest(BaseModel):
    """导入仓库请求 —— 用户提供本地绝对路径。"""
    path: str


class AskRequest(BaseModel):
    """提问请求 —— 指定仓库 ID 和问题文本。"""
    repo_id: str
    question: str


class SearchHit(BaseModel):
    """检索命中结果 —— 包含完整 Chunk 和相关性得分。"""
    chunk: Chunk
    score: float = 0.0
    source: str = ""  # 来源标记：dense / sparse / exact
