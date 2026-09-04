"""
把 ecommerce-test-automation 的固件（fixture）"借"过来用

------------------------------------------------------------------
为什么需要这个文件
------------------------------------------------------------------
AI 生成的测试代码放在**我们自己的** generated_tests/ 目录
（约定：不往 ecommerce-test-automation 里写任何文件）。

但生成的代码里会用 `page_context` 这类固件，而这些固件定义在
**对方项目根目录的 conftest.py** 里。pytest 只会从"被收集的文件"
所在目录往上找 conftest.py，所以对方那份不会被自动加载。

解决办法：在这里用 importlib 按文件路径把对方的 conftest.py 加载进来，
把里面的固件函数重新导出。**整个过程不修改对方的任何文件。**

------------------------------------------------------------------
如果对方项目不可用怎么办
------------------------------------------------------------------
下面写了兜底方案：加载不到时，自己定义一个简易版 page_context。
这样即使对方项目被挪走，这里的测试也不会"收集就报错"，
而是能明确告诉你"环境没配好"。这叫优雅降级。
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

# 对方项目根目录（与 config/settings.py 里的 TEST_PROJECT_DIR 一致）
TEST_PROJECT = Path(r"D:/PythonProjects/ecommerce-test-automation")


def _load_test_project_conftest():
    """按文件路径加载对方项目的 conftest.py，返回模块对象。"""
    if str(TEST_PROJECT) not in sys.path:
        sys.path.insert(0, str(TEST_PROJECT))

    conftest_path = TEST_PROJECT / "conftest.py"
    if not conftest_path.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location("_tp_conftest", conftest_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001 - 对方环境有问题时，降级而不是让收集阶段直接崩
        return None


_tp = _load_test_project_conftest()

if _tp is not None:
    # 把对方的固件原样导出，pytest 就能在本目录里使用它们
    page_context = _tp.page_context
    ensure_test_account = _tp.ensure_test_account
else:
    # 兜底：自己定义一个简易版 page_context，保证测试至少能被收集和执行
    @pytest.fixture(scope="function")
    def page_context(page):  # type: ignore[no-redef]
        """简易版页面固件（对方项目不可用时的兜底）。"""
        try:
            page.set_default_timeout(30000)
            page.set_default_navigation_timeout(30000)
        except Exception:  # noqa: BLE001
            pass
        yield page

    @pytest.fixture(scope="session")
    def ensure_test_account():  # type: ignore[no-redef]
        """兜底版：不做任何账号预建，只打个日志。"""
        yield


# ------------------------------------------------------------------
# Day 22：失败截图
# ------------------------------------------------------------------
# 失败截图要写到**本项目**自己的 outputs/screenshots/ 目录，
# 不能写到对方项目里（守住"不改动对方任何文件"的约定）。
#
# 这里故意不用 `from config import settings`：
# 因为生成的测试是在"对方项目的 pytest 子进程"里跑的，
# 那个子进程的 sys.path 里只有对方项目根，import 不到我们的 config。
# 所以用相对路径直接算出来最稳：
#   generated_tests/conftest.py -> parent.parent = 项目根 ai-test-agent/
SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "screenshots"


def _safe_name(nodeid: str) -> str:
    """把 pytest 的 nodeid 变成文件名安全的字符串。

    例如：
        tests/test_login.py::test_x  ->  tests_test_login.py__test_x
    """
    name = nodeid.replace("::", "__").replace("/", "_")
    return re.sub(r"[^\w.\-]", "_", name)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """用例失败时自动截图（Day 22）。

    这是 pytest 的"钩子"（hook）：pytest 每跑完一个用例都会调它。
    hookwrapper 的意思是"在 pytest 自己处理前后插一脚"，
    我们这里只需要在"调用（call）阶段失败"时额外截一张图，
    完全不影响 pytest 的正常流程。

    为什么写到 nodeid 命名的文件？
        因为后面 failure_collector 要按 nodeid 把"失败信息"和"截图"
        对应起来。命名一致，才能对得上号。
    """
    outcome = yield
    report = outcome.get_result()
    # 只在"执行阶段(call)"失败时才截图；setup/teardown 失败不算业务失败
    if report.when == "call" and report.failed:
        page = item.funcargs.get("page")
        if page is not None:
            try:
                SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
                path = SCREENSHOT_DIR / f"{_safe_name(item.nodeid)}.png"
                page.screenshot(path=str(path))
            except Exception:  # noqa: BLE001 - 截图只是辅助，失败不影响主流程
                pass
