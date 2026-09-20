import os
import tempfile

# 必须在任何 app 模块导入前生效：让测试走临时 sqlite 文件，而不是真实 Postgres。
# conftest 先于所有测试模块被 pytest 导入，因此这里的 env 设置一定早于 app.config 实例化。
_DB_PATH = os.path.join(tempfile.gettempdir(), "ep_api_test.db")
if os.path.exists(_DB_PATH):
    os.remove(_DB_PATH)
os.environ["DATABASE_URL"] = f"sqlite+pysqlite:///{_DB_PATH}"
