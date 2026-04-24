"""
OpsWiki 启动入口。

注意：Python 3.14 + Windows 下 uvicorn 的 reload 模式存在兼容性问题
（pathlib 内部重构导致 multiprocessing spawn 失败），因此默认关闭 reload。
如需热重载，请使用 Python 3.11~3.13。
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run("app.api:app", host=settings.host, port=settings.port, reload=False)
