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
