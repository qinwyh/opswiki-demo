"""
OpsWiki - 配置模块
==================
负责读取环境变量、管理全局配置项（LLM、Embedding、检索参数、服务端口等），
并提供统一的日志初始化函数。

使用 pydantic-settings 从 .env 文件自动加载配置。
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
import logging


class Settings(BaseSettings):
    """
    全局配置类。
    字段名与 .env 中的环境变量名一一对应（不区分大小写）。
    pydantic-settings 会自动从 .env 文件和系统环境变量中读取值。
    """

    # ── API 密钥 ──
    # 优先使用 QWEN_API_KEY，若未设置则回退到 DASHSCOPE_API_KEY
    qwen_api_key: str = Field(default="")
    dashscope_api_key: str = Field(default="")

    # ── LLM（大语言模型）配置 ──
    llm_model: str = "qwen3-coder-next"                                      # 模型名称
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # OpenAI 兼容端点
    llm_temperature: float = 0.2                                              # 生成温度（越低越确定）
    llm_max_tokens: int = 4096                                                # 单次最大生成 token 数

    # ── Embedding（向量嵌入）配置 ──
    embedding_provider: str = "dashscope"   # 可选值: dashscope | local（预留本地模型切换）
    embedding_model: str = "text-embedding-v3"
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_dimension: int = 1024         # 向量维度，需与所选模型输出维度一致

    # ── 检索参数 ──
    retrieval_top_k: int = 20    # 初次多路召回数量（融合前每路取这么多）
    rerank_top_k: int = 8        # 重排后最终保留的结果条数
    graph_expand_hops: int = 1   # 轻量代码图扩展跳数（0 = 关闭）

    # ── 服务配置 ──
    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: str = "./data"     # 索引、SQLite 等持久化数据的根目录
    log_level: str = "INFO"

    # pydantic-settings 的元配置：.env 文件路径与编码
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def api_key(self) -> str:
        """
        统一获取 API 密钥。
        优先级: qwen_api_key → dashscope_api_key → 环境变量 QWEN_API_KEY → DASHSCOPE_API_KEY
        """
        return (
            self.qwen_api_key
            or self.dashscope_api_key
            or os.getenv("QWEN_API_KEY", "")
            or os.getenv("DASHSCOPE_API_KEY", "")
        )

    @property
    def data_path(self) -> Path:
        """返回数据存储目录的 Path 对象，不存在时自动创建。"""
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


# 全局单例配置对象 —— 其它模块统一通过 `from app.config import settings` 引用
settings = Settings()


def setup_logging() -> None:
    """根据配置初始化全局日志格式与级别。"""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
