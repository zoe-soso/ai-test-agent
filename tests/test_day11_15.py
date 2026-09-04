"""
Days 11~15 的单元测试。

覆盖：
    Day 11  用例质量评审（规则层 + 评审闭环）
    Day 12  测试数据生成（覆盖自检 + 质量检查 + 标记展开）
    Day 13  Tool Calling（Schema 生成 + 工具注册 + 循环 + chat_with_tools）
    Day 14  Agent MVP（planner 把生成→评审→落盘串起来）

这些测试全部离线可跑（用 StubClient / FakeClient 代替真实大模型），
不花一分钱，也不依赖网络。真实 API 的连通性交给命令行 `main.py agent` 现场验证。
"""

import json
from types import SimpleNamespace

import pytest

from agent import reviewer, testdata_generator, tool_calling, planner, llm_client
from agent.models import DATA_TYPES
from agent.testcase_builder import build_login_cases
from config import settings
from tools import file_io


# =====================================================================
# 离线桩：代替真实大模型
# =====================================================================
class StubClient:
    """永远返回同一段合法用例 JSON 的假客户端。

    对 planner 来说，生成阶段拿到合法用例、评审阶段解析不出问题
    （那段 JSON 没有 issues/overall 字段），于是整条链路顺利跑完。
    """

    def __init__(self) -> None:
        cases = build_login_cases()
        self.answer = json.dumps({"cases": cases}, ensure_ascii=False)
        self.usage = SimpleNamespace(calls=0, __str__=lambda self: "（离线桩，无用量统计）")

    def chat_messages(self, messages, **kwargs):
        return self.answer

    def chat(self, prompt, **kwargs):
        return self.answer


# =====================================================================
# Day 11：测试用例质量检查
# =====================================================================
def _case(cid, name, case_type="正常", priority="P1",
          steps=None, expected="登录成功"):
    if steps is None:
        steps = ["打开页面", "点击登录"]
    return {
        "id": cid, "name": name, "case_type": case_type,
        "priority": priority, "steps": steps,
        "expected": expected,
    }


def test_rule_check_finds_duplicate_id():
    cases = [_case("TC_001", "正常登录"), _case("TC_001", "重复编号")]
    issues = reviewer.check_duplicates(cases)
    assert any(i.category == "重复" and "TC_001" in i.message for i in issues)


def test_rule_check_finds_empty_steps():
    cases = [_case("TC_002", "没步骤", steps=[])]
    issues = reviewer.check_case_quality(cases)
    assert any(i.category == "步骤" for i in issues)


def test_rule_check_flags_missing_exception_type():
    cases = [
        _case("TC_010", "正常1", "正常"),
        _case("TC_011", "正常2", "正常"),
        _case("TC_012", "边界1", "边界"),
    ]
    issues = reviewer.check_distribution(cases)
    assert any("异常" in i.message for i in issues)


def test_rule_check_passes_good_cases():
    #  realistic: 三类齐全、步骤各不相同、且有一条 P0 主流程
    cases = [
        _case("TC_020", "正常登录", "正常", "P0",
              steps=["输入正确账号", "输入正确密码", "点击登录"]),
        _case("TC_021", "密码错误", "异常", "P1",
              steps=["输入正确账号", "输入错误密码", "点击登录"]),
        _case("TC_022", "密码超长", "边界", "P3",
              steps=["输入正确账号", "输入500位密码", "点击登录"]),
    ]
    issues = reviewer.check_by_rules(cases)
    assert issues == [], f"好用例不应有问题，但查到：{issues}"


def test_review_and_revise_runs_offline():
    # 用合法用例 + 桩客户端，整条"生成→评审→修改"链路不应抛异常，
    # 且返回的用例数量与输入一致（桩客户端不会真改内容）。
    cases = build_login_cases()
    report = reviewer.ReviewReport(feature="登录")
    final, review, rounds = reviewer.review_and_revise(
        "用户登录功能", cases, client=StubClient(), max_rounds=1, use_llm=True
    )
    assert len(final) == len(cases)
    assert rounds == 0  # 合法用例没有错误级问题，不会进入修改轮


# =====================================================================
# Day 12：测试数据生成
# =====================================================================
def test_data_coverage_detects_missing_type():
    present = [
        {"id": "TD_1", "name": "a", "data_type": "正确数据", "fields": {"x": "1"}},
        {"id": "TD_2", "name": "b", "data_type": "错误数据", "fields": {"x": "2"}},
    ]
    missing = testdata_generator.TestDataGenerator(client=StubClient()).check_coverage(present)
    # 六类里只给了 2 类，剩下 4 类应被判缺
    assert len(missing) == len(DATA_TYPES) - 2


def test_data_quality_flags_placeholder():
    data = [{
        "id": "TD_3", "name": "占位符", "data_type": "正确数据",
        "fields": {"password": "正确的密码"},
    }]
    problems = testdata_generator.TestDataGenerator(client=StubClient()).quality_check(data)
    assert any("占位符" in p for p in problems)


def test_data_quality_flags_short_overlong():
    data = [{
        "id": "TD_4", "name": "超长不够", "data_type": "超长数据",
        "fields": {"password": "x" * 10},  # 远小于 MIN_OVERLONG_LENGTH=100
    }]
    problems = testdata_generator.TestDataGenerator(client=StubClient()).quality_check(data)
    assert any("超长数据" in p for p in problems)


def test_expand_markers_produces_real_length():
    assert testdata_generator.expand_markers("<<LONG:500:A>>") == "A" * 500
    assert testdata_generator.expand_markers(123) == 123  # 非字符串原样返回


# =====================================================================
# Day 13：Tool Calling
# =====================================================================
def test_parse_docstring():
    def f(a: int, b: str) -> str:
        """执行某个动作。

        Args:
            a: 第一个参数
            b: 第二个参数
        """

    desc, params = tool_calling.parse_docstring(f.__doc__)
    assert desc == "执行某个动作。"
    assert params == {"a": "第一个参数", "b": "第二个参数"}


def test_build_schema_types_and_required():
    def f(target: str, timeout: int = 300) -> str:
        """执行某个动作。

        Args:
            target: 目标路径
            timeout: 超时秒数
        """

    schema = tool_calling.build_schema(f)
    assert schema["type"] == "object"
    assert schema["properties"]["target"]["type"] == "string"
    assert schema["properties"]["timeout"]["type"] == "integer"
    # timeout 有默认值，不应在 required 里
    assert schema["required"] == ["target"]


def test_tool_registry_call_and_unknown():
    reg = tool_calling.ToolRegistry()

    @reg.tool
    def add(a: int, b: int) -> str:
        """两数相加。

        Args:
            a: 第一个数
            b: 第二个数
        """
        return str(a + b)

    # 正常调用
    out = reg.call("add", {"a": 2, "b": 3})
    assert out == "5"

    # 未知工具：返回错误提示，而非抛异常（Agent 要能自我纠正）
    err = reg.call("nope", {})
    assert err.startswith("[错误]")


def test_run_tool_loop_executes_then_finishes():
    from agent.tool_calling import AssistantMessage, ToolCall

    reg = tool_calling.ToolRegistry()

    @reg.tool
    def add(a: int, b: int) -> str:
        """两数相加。

        Args:
            a: 第一个数
            b: 第二个数
        """
        return str(a + b)

    # 第一轮：模型要调 add(2,3)；第二轮：给出最终答复
    responses = iter([
        AssistantMessage(content="", tool_calls=[
            ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})
        ]),
        AssistantMessage(content="结果是 5", tool_calls=[]),
    ])

    class FakeClient:
        def chat_with_tools(self, messages, tools):
            return next(responses)

    result = tool_calling.run_tool_loop(FakeClient(), [], reg, max_iterations=5)
    assert result.iterations == 2
    assert result.answer == "结果是 5"
    assert result.tool_names == ["add"]
    assert result.steps[0].result == "5"


def test_chat_with_tools_mock_runs_offline():
    # Mock 模式下不应真实调用，也不应报错；
    # 现在 Mock 会按"关键词剧本"返回工具调用（这样 Day 13/26 的
    # 工具循环在离线时也能真正跑通）。这里验证它返回的是合法的
    # AssistantMessage，且对"生成用例"类指令会给出工具调用。
    client = llm_client.LLMClient(api_key=None, mock_mode="true")
    msg = client.chat_with_tools(
        [{"role": "user", "content": "帮我把登录功能的测试用例生成并保存"}], []
    )
    assert isinstance(msg, tool_calling.AssistantMessage)
    assert msg.wants_tools is True
    assert any(call.name == "generate_testcase" for call in msg.tool_calls)


# =====================================================================
# Day 14：Agent MVP（planner）
# =====================================================================
def test_planner_runs_pipeline_and_saves(tmp_path):
    agent = planner.TestCaseAgent(
        client=StubClient(), auto_fix=True, use_llm=True, max_repairs=1
    )
    cases_out = tmp_path / "cases.yaml"
    review_out = tmp_path / "review.yaml"
    result = agent.run(
        "用户登录功能", cases_path=cases_out, review_path=review_out
    )

    assert (tmp_path / "cases.yaml").exists()
    assert (tmp_path / "review.yaml").exists()
    assert len(result.suite["cases"]) == len(build_login_cases())
    assert result.revise_rounds == 0
    assert result.review.passed  # 合法用例应通过评审


def test_planner_rule_only_skips_llm():
    # rule_only 模式：评审阶段不调 LLM，但仍应正常跑完
    agent = planner.TestCaseAgent(client=StubClient(), use_llm=False, auto_fix=False)
    result = agent.run("用户登录功能")
    assert result.suite["cases"]
    # 规则层对合法登录用例不会判错
    assert result.review.passed


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
