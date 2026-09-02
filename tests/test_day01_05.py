"""
Day 1~5 的冒烟测试

为什么现在就写测试？
    1. 验证新虚拟环境里的 pytest 确实能用；
    2. 给你自己一个"安全网"——后面改代码时敢改；
    3. 面试时能说"我的 AI 项目本身也有单元测试"，这是加分项。

运行：
    venv\\Scripts\\python.exe -m pytest tests/ -v
"""

from __future__ import annotations

import pytest

from agent import llm_client, requirement_reader, testcase_builder
from config import settings
from tools import file_io
from tools.exceptions import RequirementError


# ---------------- Day 1：环境 ----------------
def test_python_version_is_312():
    import sys

    assert sys.version_info[:2] == (3, 12)


def test_test_project_is_readonly_reference():
    """关联的测试项目必须存在，但我们只读它，不改它。"""
    assert settings.TEST_PROJECT_DIR.exists()
    assert settings.TEST_PROJECT_PYTHON.exists()


# ---------------- Day 2：数据结构与函数 ----------------
def test_dedupe_steps_keeps_order():
    steps = ["打开首页", "点击登录", "打开首页", "输入密码"]
    assert testcase_builder.dedupe_steps(steps) == ["打开首页", "点击登录", "输入密码"]


def test_build_login_cases_shape():
    cases = testcase_builder.build_login_cases()
    assert len(cases) == 8

    required_keys = {"id", "name", "case_type", "priority", "steps", "expected"}
    for case in cases:
        assert required_keys.issubset(case.keys()), f"字段缺失：{case}"
        assert case["case_type"] in ("正常", "异常", "边界")
        assert case["priority"].startswith("P")
        assert isinstance(case["steps"], list) and case["steps"]


def test_build_test_suite_raises_on_unknown_feature():
    with pytest.raises(ValueError):
        testcase_builder.build_test_suite("一个不存在的功能")


def test_summarize_counts():
    suite = testcase_builder.build_test_suite("用户登录功能")
    stat = testcase_builder.summarize(suite)
    assert stat["总数"] == 8
    assert stat["类型-正常"] == 1
    assert stat["类型-异常"] == 4
    assert stat["类型-边界"] == 3


# ---------------- Day 3：文件读写 ----------------
def test_yaml_roundtrip(tmp_path):
    """写入再读回来，内容必须一致（顺便验证中文不乱码）。"""
    suite = testcase_builder.build_test_suite("用户登录功能")
    target = tmp_path / "sub" / "testcases.yaml"  # 顺便验证会自动建父目录

    file_io.write_yaml(target, suite)
    loaded = file_io.read_yaml(target)

    assert loaded == suite
    assert loaded["cases"][0]["name"] == "使用已注册账号正常登录"


def test_json_roundtrip(tmp_path):
    data = {"name": "正常登录", "steps": ["输入账号"], "expected": "登录成功"}
    target = tmp_path / "case.json"

    file_io.write_json(target, data)

    assert file_io.read_json(target) == data
    assert "正常登录" in file_io.read_text(target)  # 中文没被转义


def test_requirement_parsing():
    requirement = requirement_reader.load(settings.DATA_DIR / "requirement.txt")
    assert requirement.feature == "用户登录功能"
    assert "automationexercise.com" in requirement.description


def test_requirement_empty_raises():
    with pytest.raises(RequirementError):
        requirement_reader.parse_text("   \n\n  ", source="<test>")


def test_requirement_missing_file_raises():
    with pytest.raises(RequirementError):
        requirement_reader.load("data/绝对不存在.txt")


# ---------------- Day 4：异常 ----------------
def test_exception_hierarchy():
    from tools.exceptions import AgentError, LLMError, LLMNotConfiguredError

    assert issubclass(LLMNotConfiguredError, LLMError)
    assert issubclass(LLMError, AgentError)  # 能被 `except AgentError` 一把兜住


def test_exception_carries_context():
    from tools.exceptions import LLMNotConfiguredError

    exc = LLMNotConfiguredError("没配 Key", model="deepseek-chat")
    assert exc.context["model"] == "deepseek-chat"
    assert "model=deepseek-chat" in str(exc)
    assert exc.user_message  # 必须有给人看的提示


def test_logger_writes_file():
    from tools.logger import get_logger

    logger = get_logger("test")
    logger.info("这是一条测试日志：中文必须正常")

    log_file = settings.LOG_DIR / "agent.log"
    assert log_file.exists()

    content = log_file.read_text(encoding="utf-8")  # 能用 utf-8 读通就说明没乱码
    assert "这是一条测试日志" in content


# ---------------- Day 5：LLM 客户端 ----------------
def test_mock_mode_is_default_without_key():
    client = llm_client.LLMClient(api_key=None, mock_mode="auto")
    assert client.is_mock is True


def test_mock_mode_can_be_forced_off():
    client = llm_client.LLMClient(api_key=None, mock_mode="false")
    assert client.is_mock is False

    with pytest.raises(Exception) as exc_info:
        client.chat("你好")
    assert "LLM_API_KEY" in str(exc_info.value)


def test_mock_chat_returns_answer():
    client = llm_client.LLMClient(api_key=None, mock_mode="true")
    answer = client.chat("请帮我设计登录功能的测试用例。")
    assert "TC_LOGIN_001" in answer


def test_strip_code_fence():
    raw = '```json\n{"cases": []}\n```'
    assert llm_client._strip_code_fence(raw) == '{"cases": []}'


def test_ensure_dirs_idempotent():
    """重复调用不应该报错。"""
    settings.ensure_dirs()
    settings.ensure_dirs()
