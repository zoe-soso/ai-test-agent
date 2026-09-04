"""
Days 16~20 的单元测试。

覆盖：
    Day 16/17  代码生成规则（页面对象解析、代码提取、方法真实性检查）
    Day 18     代码检查（语法 / POM / 固件 / 是否有先打开站点）
    Day 19     Human-in-the-loop（人工确认）
    Day 20     pytest 执行器（退出码解析、统计行解析）

全部离线可跑，不调用真实大模型，也不真的执行浏览器。
真实链路的验证交给 `main.py code --run`（会真实调用 API + 真实跑 pytest）。
"""

import builtins
import pytest

from agent import code_generator
from tools import test_runner, human
from config import settings


# =====================================================================
# 一份"理想"的生成代码（应该通过所有检查）
# =====================================================================
GOOD_CODE = '''
import pytest
from pages.login_page import LoginPage
from utils.config_reader import load_config

config = load_config()

@pytest.mark.parametrize("email,password,expected", [("a@b.com", "123456", "zoe")])
def test_login_ok(page_context, email, password, expected):
    login_page = LoginPage(page_context)
    login_page.open(config["base_url"])
    login_page.open_login_page()
    login_page.login(email, password)
    assert expected in login_page.get_login_user()
'''


# =====================================================================
# Day 16：功能 -> 页面对象 的对应
# =====================================================================
def test_resolve_page_maps_login():
    page_class, module, _hint = code_generator.resolve_page("用户登录功能")
    assert page_class == "LoginPage"
    assert module == "pages.login_page"


def test_resolve_page_maps_cart():
    page_class, module, _hint = code_generator.resolve_page("购物车结算")
    assert page_class == "CartPage"
    assert module == "pages.cart_page"


def test_resolve_page_falls_back():
    page_class, _module, _hint = code_generator.resolve_page("某个没见过的功能")
    assert page_class == "LoginPage"  # 兜底


# =====================================================================
# Day 17：从模型回复里提取代码
# =====================================================================
def test_extract_python_code_from_fence():
    raw = '好的，代码如下：\n```python\nprint("hi")\n```\n希望能帮到你'
    assert code_generator.extract_python_code(raw) == 'print("hi")'


def test_extract_python_code_without_fence():
    assert code_generator.extract_python_code("x = 1") == "x = 1"


# =====================================================================
# Day 18：代码检查
# =====================================================================
def test_validate_code_accepts_good_code():
    issues = code_generator.validate_code(GOOD_CODE)
    assert issues == [], f"好代码不该有问题，但查到：{issues}"


def test_validate_code_catches_syntax_error():
    issues = code_generator.validate_code("def test_x(:\n    pass")
    assert any("语法错误" in i for i in issues)


def test_validate_code_catches_raw_playwright_api():
    bad = GOOD_CODE + "\n    page.locator('a').click()"
    issues = code_generator.validate_code(bad)
    assert any("POM" in i for i in issues)


def test_validate_code_catches_missing_fixture():
    bad = GOOD_CODE.replace("def test_login_ok(page_context, email, password, expected):",
                            "def test_login_ok():")
    issues = code_generator.validate_code(bad)
    assert any("page_context" in i for i in issues)


def test_validate_code_catches_missing_page_import():
    bad = GOOD_CODE.replace("from pages.login_page import LoginPage", "")
    issues = code_generator.validate_code(bad)
    assert any("没有导入页面对象" in i for i in issues)


def test_validate_code_catches_hallucinated_method():
    # 模型最爱编造看起来合理但根本不存在的方法名
    bad = GOOD_CODE.replace("login_page.login(email, password)",
                            "login_page.fill_email(email)")
    issues = code_generator.validate_code(bad)
    assert any("不存在的方法" in i for i in issues)


def test_validate_code_catches_missing_site_open():
    # 不先打开站点首页，后面的点击全部会超时
    bad = GOOD_CODE.replace('login_page.open(config["base_url"])', "")
    issues = code_generator.validate_code(bad)
    assert any("打开被测站点" in i for i in issues)


def test_load_page_methods_reads_real_source():
    # 从目标项目源码静态解析出真实方法清单（需要目标项目存在）
    if not settings.TEST_PROJECT_DIR.exists():
        pytest.skip("找不到关联的测试项目，跳过")
    methods = code_generator.load_page_methods("pages.login_page")
    assert "login" in methods
    assert "open" in methods           # 来自 BasePage
    assert "fill_email" not in methods  # 编造的方法不该在里面


def test_generated_code_dataclass_ok_flag():
    from agent.code_generator import GeneratedCode
    empty = GeneratedCode(feature="x", case_id="TC_1", case_name="n")
    assert empty.ok is False
    ok = GeneratedCode(feature="x", case_id="TC_1", case_name="n", code="x = 1")
    assert ok.ok is True


# =====================================================================
# Day 19：Human-in-the-loop
# =====================================================================
def test_ask_yes_no_yes(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *_: "y")
    assert human.ask_yes_no("执行吗") is True


def test_ask_yes_no_no(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *_: "n")
    assert human.ask_yes_no("执行吗") is False


def test_ask_yes_no_default_is_no(monkeypatch):
    # 直接回车 = 不同意。这是安全默认值，避免随手回车就执行代码
    monkeypatch.setattr(builtins, "input", lambda *_: "")
    assert human.ask_yes_no("执行吗") is False


def test_ask_yes_no_retries_on_garbage(monkeypatch):
    answers = iter(["也许吧", "y"])
    monkeypatch.setattr(builtins, "input", lambda *_: next(answers))
    assert human.ask_yes_no("执行吗") is True


# =====================================================================
# Day 20：pytest 执行器（只测解析，不真跑）
# =====================================================================
def test_parse_output_extracts_counts_and_failures():
    output = (
        "FAILED tests/test_login.py::test_a - AssertionError: expected zoe\n"
        "1 failed, 2 passed in 3.45s\n"
    )
    result = test_runner.TestRunResult(command=[])
    test_runner._parse_output(output, result)

    assert result.passed == 2
    assert result.failed == 1
    assert result.summary_line == "1 failed, 2 passed in 3.45s"
    assert len(result.failed_tests) == 1
    assert "test_a" in result.failed_tests[0]


def test_parse_output_all_passed():
    output = "3 passed in 1.20s\n"
    result = test_runner.TestRunResult(command=[])
    test_runner._parse_output(output, result)
    assert result.passed == 3
    assert result.failed == 0
    assert result.summary_line == "3 passed in 1.20s"
    # returncode 默认就是 0（成功），所以 success 为 True
    assert result.success is True


def test_exit_code_meanings():
    result = test_runner.TestRunResult(command=[], returncode=1)
    assert result.exit_meaning == "有用例失败"
    assert result.success is False

    result = test_runner.TestRunResult(command=[], returncode=0)
    assert result.exit_meaning == "全部通过"
    assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
