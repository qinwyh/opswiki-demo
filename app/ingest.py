"""
OpsWiki - 数据摄入模块（Ingest）
================================
负责仓库扫描、文件过滤、结构化解析与分块。

核心能力：
- scan_repository：遍历目录树，过滤无关目录，收集 .py / .md 文件
- parse_markdown：按 Markdown 标题层级拆分为 section chunks
- parse_python：使用 Python AST 将源码拆分为 file_summary / class / function / method chunks
- ingest_repository：组合以上步骤的完整摄入流水线

设计原则：
- 不使用固定字数切块，而是按代码结构（AST 节点）和文档结构（标题层级）做语义分块
- 每个 chunk 携带丰富元数据（路径、行号、签名、docstring、调用关系等），支持精准溯源
"""

from __future__ import annotations
import ast
import hashlib
import logging
import os
from pathlib import Path
from typing import Generator

from markdown_it import MarkdownIt

from app.schemas import Chunk, ChunkType, CodeEdge

logger = logging.getLogger(__name__)

# 需要跳过的目录集合 —— 包括版本控制、虚拟环境、缓存、构建产物等
IGNORED_DIRS = {
    ".git", ".venv", "__pycache__", "node_modules", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".tox", ".eggs", ".idea", ".vscode",
    "venv", "env", ".env", ".svn", ".hg",
}

# 当前 MVP 支持的文件扩展名（后续可扩展 .ts / .java / .go 等）
SUPPORTED_EXTENSIONS = {".py", ".md"}


def _make_id(*parts: str) -> str:
    """根据多个字符串片段生成一个短 hash ID（16 字符），用作 chunk 唯一标识。"""
    return hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════
#  仓库扫描
# ══════════════════════════════════════════════════════════════

def scan_repository(repo_path: Path) -> list[Path]:
    """
    递归扫描仓库目录，返回所有受支持的文件路径列表。

    安全策略：
    - 跳过 IGNORED_DIRS 中列出的目录（如 .git、node_modules）
    - 跳过以 . 开头的隐藏目录
    - 仅收集 SUPPORTED_EXTENSIONS 中的文件类型
    """
    files: list[Path] = []
    for root, dirs, filenames in os.walk(repo_path):
        # 原地修改 dirs 列表以阻止 os.walk 进入被忽略的子目录
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
        for fn in filenames:
            fp = Path(root) / fn
            if fp.suffix in SUPPORTED_EXTENSIONS:
                files.append(fp)
    return sorted(files)


# ══════════════════════════════════════════════════════════════
#  Markdown 解析与分块
# ══════════════════════════════════════════════════════════════

def parse_markdown(file_path: Path, repo_path: Path, repo_id: str) -> list[Chunk]:
    """
    将 Markdown 文件按标题层级拆分为多个 section chunks。

    分块策略：
    1. 遇到 # / ## / ### 等标题行时，创建新的 section
    2. 每个 section 记录标题、标题层级路径（如 "安装 > 配置"）、起止行号
    3. 标题前的内容归入 "(preamble)" 伪章节
    4. 过短的 section（< 5 字符）被丢弃

    返回：Chunk 列表，每个 chunk 的 content 包含标题 + 正文
    """
    rel = file_path.relative_to(repo_path).as_posix()  # 相对路径（POSIX 风格）
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"无法读取文件 {file_path}: {e}")
        return []

    lines = text.split("\n")

    # heading_stack 用于维护当前标题的层级路径
    # 例如当前在 ## 配置 下的 ### 环境变量，则 stack = ["配置", "环境变量"]
    heading_stack: list[str] = []
    sections: list[dict] = []
    current_section: dict | None = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            # ── 遇到标题行：保存上一个 section，开始新 section ──
            if current_section:
                sections.append(current_section)

            # 计算标题级别（# 的数量）
            hashes = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()

            # 更新标题栈：弹出所有 >= 当前级别的标题，再压入当前标题
            while len(heading_stack) >= hashes:
                heading_stack.pop() if heading_stack else None
            heading_stack.append(title)

            current_section = {
                "title": title,
                "heading_path": " > ".join(heading_stack),
                "start_line": i + 1,   # 行号从 1 开始
                "lines": [],
                "level": hashes,
            }
        else:
            # ── 普通内容行：追加到当前 section ──
            if current_section is None:
                # 标题前的内容归入 preamble
                current_section = {
                    "title": "(preamble)",
                    "heading_path": "(preamble)",
                    "start_line": 1,
                    "lines": [],
                    "level": 0,
                }
            current_section["lines"].append(line)

    # 别忘了保存最后一个 section
    if current_section:
        sections.append(current_section)

    # ── 将 sections 转为 Chunk 对象 ──
    chunks: list[Chunk] = []
    for sec in sections:
        content = "\n".join(sec["lines"]).strip()
        # 跳过空的 preamble
        if not content and sec["title"] == "(preamble)":
            continue
        # 构造用于 embedding 的文本：标题 + 正文
        text_for_embed = f"# {sec['title']}\n\n{content}" if sec["title"] != "(preamble)" else content
        # 过滤过短的内容
        if len(text_for_embed.strip()) < 5:
            continue

        end_line = sec["start_line"] + len(sec["lines"])
        chunks.append(Chunk(
            chunk_id=_make_id(repo_id, rel, sec["title"], str(sec["start_line"])),
            repo_id=repo_id,
            rel_path=rel,
            language="markdown",
            chunk_type=ChunkType.MD_SECTION,
            symbol_name=sec["title"],
            start_line=sec["start_line"],
            end_line=end_line,
            content=text_for_embed,
            heading_path=sec["heading_path"],
        ))
    return chunks


# ══════════════════════════════════════════════════════════════
#  Python AST 解析与分块
# ══════════════════════════════════════════════════════════════

def parse_python(file_path: Path, repo_path: Path, repo_id: str) -> tuple[list[Chunk], list[CodeEdge]]:
    """
    使用 Python 标准库 ast 模块解析 .py 文件，生成结构化 chunks 和代码关系边。

    分块策略（不使用固定字数切块）：
    1. file_summary chunk：整个文件的概览（imports、顶层定义列表、模块 docstring）
    2. class chunk：每个类定义（含 docstring 和方法列表）
    3. function chunk：每个顶层函数
    4. method chunk：类中的每个方法

    同时提取代码关系边（CodeEdge）：
    - file → symbol（contains 关系）
    - file → module（imports 关系）
    - function → callee（calls 关系，轻量近似提取）

    返回：(chunks列表, edges列表)
    """
    rel = file_path.relative_to(repo_path).as_posix()
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"无法读取文件 {file_path}: {e}")
        return [], []

    lines = source.split("\n")

    # ── 尝试 AST 解析，语法错误时降级为整文件 chunk ──
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as e:
        logger.warning(f"语法错误 {file_path}: {e}")
        return [Chunk(
            chunk_id=_make_id(repo_id, rel, "module"),
            repo_id=repo_id, rel_path=rel, language="python",
            chunk_type=ChunkType.MODULE, symbol_name=rel,
            start_line=1, end_line=len(lines),
            content=source[:3000], raw_code=source[:3000],
        )], []

    chunks: list[Chunk] = []
    edges: list[CodeEdge] = []

    # ── 1. 提取模块级 import 信息 ──
    mod_imports: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            # import os, sys → ["os", "sys"]
            for alias in node.names:
                mod_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # from pathlib import Path → "pathlib.Path"
            module = node.module or ""
            for alias in node.names:
                mod_imports.append(f"{module}.{alias.name}")

    # ── 2. 收集顶层定义名称（用于文件摘要） ──
    top_level_names = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level_names.append(f"def {node.name}")
        elif isinstance(node, ast.ClassDef):
            top_level_names.append(f"class {node.name}")

    # ── 3. 生成 file_summary chunk ──
    summary_content = (
        f"File: {rel}\n"
        f"Imports: {', '.join(mod_imports[:20])}\n"
        f"Defines: {', '.join(top_level_names[:30])}"
    )
    module_docstring = ast.get_docstring(tree) or ""
    if module_docstring:
        summary_content = f"{module_docstring}\n\n{summary_content}"

    file_summary = Chunk(
        chunk_id=_make_id(repo_id, rel, "file_summary"),
        repo_id=repo_id, rel_path=rel, language="python",
        chunk_type=ChunkType.FILE_SUMMARY, symbol_name=rel,
        start_line=1, end_line=len(lines),
        imports=mod_imports, content=summary_content, docstring=module_docstring,
    )
    chunks.append(file_summary)

    # ── 4. 记录 import 关系边 ──
    for imp in mod_imports:
        edges.append(CodeEdge(source=rel, target=imp, relation="imports"))

    # ── 辅助函数 ──

    def _get_source(node: ast.AST) -> str:
        """获取 AST 节点对应的源码文本。"""
        try:
            return ast.get_source_segment(source, node) or ""
        except Exception:
            sl = getattr(node, "lineno", 1) - 1
            el = getattr(node, "end_lineno", sl + 1)
            return "\n".join(lines[sl:el])

    def _extract_calls(node: ast.AST) -> list[str]:
        """
        轻量提取函数体中的调用目标名称。
        通过遍历 AST 找所有 Call 节点，取函数名或属性名。
        注意：这是近似提取，不做完整的作用域解析。
        """
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    # 直接调用：foo()
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # 属性调用：obj.method()
                    calls.append(child.func.attr)
        return list(set(calls))[:20]  # 去重，限制数量

    def _get_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """
        从 AST 节点构造函数签名字符串。
        例如：def process(data: list[str], config: Config) -> Result
        """
        args = []
        for a in node.args.args:
            ann = ""
            if a.annotation:
                try:
                    ann = ": " + ast.unparse(a.annotation)
                except Exception:
                    pass
            args.append(f"{a.arg}{ann}")
        ret = ""
        if node.returns:
            try:
                ret = " -> " + ast.unparse(node.returns)
            except Exception:
                pass
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)}){ret}"

    def process_function(node: ast.FunctionDef | ast.AsyncFunctionDef, parent_class: str = ""):
        """
        处理单个函数/方法节点：生成对应的 Chunk 和关系边。

        参数：
            node: AST 函数节点
            parent_class: 如果是类方法，传入类名；顶层函数则为空
        """
        raw = _get_source(node)
        calls = _extract_calls(node)
        sig = _get_signature(node)
        doc = ast.get_docstring(node) or ""

        # 构造完全限定名：ClassName.method_name 或 function_name
        qname = f"{parent_class}.{node.name}" if parent_class else node.name
        ct = ChunkType.METHOD if parent_class else ChunkType.FUNCTION

        # 组装用于 embedding 的内容：签名 + docstring + 源码
        content_parts = [sig]
        if doc:
            content_parts.append(f'"""{doc}"""')
        content_parts.append(raw[:2000])

        chunk = Chunk(
            chunk_id=_make_id(repo_id, rel, qname),
            repo_id=repo_id, rel_path=rel, language="python",
            chunk_type=ct, symbol_name=node.name,
            qualified_name=qname, parent_class=parent_class,
            signature=sig, docstring=doc,
            start_line=node.lineno, end_line=node.end_lineno or node.lineno,
            imports=[], outgoing_calls=calls,
            raw_code=raw[:3000], content="\n".join(content_parts),
        )
        chunks.append(chunk)

        # 记录关系边：文件包含此符号、此符号调用了哪些其他符号
        edges.append(CodeEdge(source=rel, target=qname, relation="contains"))
        for c in calls:
            edges.append(CodeEdge(source=qname, target=c, relation="calls"))

    # ── 5. 遍历顶层节点，处理函数和类 ──
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # 顶层函数
            process_function(node)
        elif isinstance(node, ast.ClassDef):
            # ── 处理类定义 ──
            doc = ast.get_docstring(node) or ""
            raw = _get_source(node)

            # 类 chunk 的内容：类名 + docstring + 方法列表
            methods = [
                n.name for n in ast.iter_child_nodes(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            class_content = f"class {node.name}\n"
            if doc:
                class_content += f'"""{doc}"""\n'
            class_content += f"Methods: {', '.join(methods)}"

            chunks.append(Chunk(
                chunk_id=_make_id(repo_id, rel, node.name),
                repo_id=repo_id, rel_path=rel, language="python",
                chunk_type=ChunkType.CLASS, symbol_name=node.name,
                qualified_name=node.name, docstring=doc,
                start_line=node.lineno, end_line=node.end_lineno or node.lineno,
                raw_code=raw[:4000], content=class_content,
            ))
            edges.append(CodeEdge(source=rel, target=node.name, relation="contains"))

            # 递归处理类中的每个方法
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    process_function(child, parent_class=node.name)

    return chunks, edges


# ══════════════════════════════════════════════════════════════
#  完整摄入流水线
# ══════════════════════════════════════════════════════════════

def ingest_repository(repo_path: Path, repo_id: str) -> tuple[list[Chunk], list[CodeEdge]]:
    """
    完整的仓库摄入流程：扫描文件 → 按类型解析 → 生成 chunks 和 edges。

    参数：
        repo_path: 仓库根目录绝对路径
        repo_id: 仓库唯一标识

    返回：(所有 chunks, 所有 code edges)
    """
    files = scan_repository(repo_path)
    all_chunks: list[Chunk] = []
    all_edges: list[CodeEdge] = []

    for fp in files:
        if fp.suffix == ".md":
            all_chunks.extend(parse_markdown(fp, repo_path, repo_id))
        elif fp.suffix == ".py":
            chunks, edges = parse_python(fp, repo_path, repo_id)
            all_chunks.extend(chunks)
            all_edges.extend(edges)

    logger.info(f"摄入完成: {len(files)} 个文件 → {len(all_chunks)} 个分块, {len(all_edges)} 条关系边")
    return all_chunks, all_edges
