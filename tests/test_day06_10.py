"""
Day 6~10 单元测试。

覆盖新增的核心逻辑：脏 JSON 解析、结构校验、设计方法映射、
Prompt 打分、评测指标、Mock 模型、生成模块与覆盖自检。

全部离线、不调真实 API。Mock 用固定 style 保证确定性。
"""

import json

import pytest

from agent import json_utils, llm_client, prompt_lab, structured, testcase_generator, validator
from agent.models import DESIGN_METHODS, TestCase as TC
from agent.mock_llm import MockLLM
from eval import run_eval


# ----------------------------------------------------------------------
# 1. 脏输出解析（Day 7）
# ----------------------------------------------------------------------
def test_extract_json_strips_fence():
    text = '```json\n{"feature": "x", "cases": []}\n```'
    assert json_utils.extract_json(text) == {"feature": "x", "cases": []}


def test_extract_json_strips_chatty_prefix():
    text = '好的，这是结果：\n{"feature": "x", "cases": []}\n希望对你有帮助'
    assert json_utils.extract_json(text) == {"feature": "x", "cases": []}


def test_extract_json_raises_on_no_json():
    with pytest.raises(Exception):
        json_utils.extract_json("这里没有任何 JSON 结构")


def test_extract_cases():
    data = {"feature": "登录", "cases": [{"id": "TC_1"}]}
    assert json_utils.extract_cases(data) == [{"id": "TC_1"}]


# ----------------------------------------------------------------------
# 2. 结构校验（Day 7）+ 设计方法映射（Day 10）
# ----------------------------------------------------------------------
def test_validate_good_case():
    case, issue = validator.validate_case(
        {
            "id": "TC_1", "name": "正常登录", "case_type": "正常",
            "priority": "P0", "steps": ["打开首页"], "expected": "登录成功",
            "design_method": "场景法",
        }
    )
    assert case is not None
    assert issue.ok
    assert case["design_method"] == "场景法"


def test_validate_missing_expected_dropped():
    case, issue = validator.validate_case(
        {"id": "TC_2", "name": "无预期", "case_type": "正常", "steps": ["a"]}
    )
    assert case is None
    assert any("expected" in e for e in issue.errors)


def test_validate_invalid_case_type_dropped():
    case, _ = validator.validate_case(
        {"id": "TC_3", "name": "性能", "case_type": "性能测试",
         "priority": "P0", "steps": ["a"], "expected": "ok"}
    )
    assert case is None


def test_validate_priority_overflow_clamped():
    case, issue = validator.validate_case(
        {"id": "TC_4", "name": "越界优先级", "case_type": "异常",
         "priority": "P9", "steps": ["a"], "expected": "提示"}
    )
    assert case is not None
    assert case["priority"] == "P3"
    assert issue.warnings


def test_validate_design_method_alias():
    case, _ = validator.validate_case(
        {"id": "TC_5", "name": "等价类", "case_type": "正常",
         "priority": "P1", "steps": ["a"], "expected": "ok",
         "design_method": "等价类"}
    )
    assert case["design_method"] == "等价类划分"


def test_validate_unknown_design_method_is_unlabeled():
    case, issue = validator.validate_case(
        {"id": "TC_6", "name": "乱标方法", "case_type": "异常",
         "priority": "P2", "steps": ["a"], "expected": "ok",
         "design_method": "玄学测试法"}
    )
    assert case["design_method"] == "未标注"
    assert issue.warnings  # 只是 warning，不丢用例


def test_validate_cases_dedup():
    passed, _, failed = validator.validate_cases(
        [
            {"id": "TC_X", "name": "a", "case_type": "正常", "priority": "P0",
             "steps": ["s"], "expected": "e"},
            {"id": "TC_X", "name": "b", "case_type": "异常", "priority": "P1",
             "steps": ["s"], "expected": "e"},
        ]
    )
    assert len(passed) == 2
    assert any("_DUP" in c["id"] for c in passed)
    assert failed == []  # 重复 id 是 warning，不进 failed


# ----------------------------------------------------------------------
# 3. Prompt 打分（Day 6）
# ----------------------------------------------------------------------
def test_prompt_quick_score_naive_vs_engineered():
    naive = "1. 正常登录\n   验证能正常登录\n2. 错误密码\n   验证系统会提示"
    engine = ('TC_LOGIN_001 name 正常登录 case_type 正常 priority P0\n'
              'steps 预期 "Logged in as 用户名" 边界 异常')
    s_naive = prompt_lab.quick_score(naive)
    s_eng = prompt_lab.quick_score(engine)
    assert s_naive["用例条数"] == 2
    assert s_eng["覆盖边界场景"] is True
    assert s_eng["标了优先级"] is True


# ----------------------------------------------------------------------
# 4. 评测指标（Day 7 / Day 10）
# ----------------------------------------------------------------------
def test_compute_method_coverage_full():
    cases: list[TC] = [
        {"id": f"T{i}", "name": "n", "case_type": "正常", "priority": "P0",
         "steps": ["s"], "expected": "e", "design_method": m}
        for i, m in enumerate(DESIGN_METHODS)
    ]
    rate, used = run_eval.compute_method_coverage(cases)
    assert rate == 1.0
    assert len(used) == len(DESIGN_METHODS)


def test_compute_method_coverage_excludes_unlabeled():
    cases: list[TC] = [
        {"id": "T1", "name": "n", "case_type": "正常", "priority": "P0",
         "steps": ["s"], "expected": "e", "design_method": "未标注"},
    ]
    rate, used = run_eval.compute_method_coverage(cases)
    assert rate == 0.0
    assert used == []


def test_compute_coverage_hit_and_miss():
    cases: list[TC] = [
        {"id": "T1", "name": "正常登录", "case_type": "正常", "priority": "P0",
         "steps": ["输入正确密码"], "expected": "登录成功"},
    ]
    scenarios = [
        {"name": "成功登录", "keywords": ["登录成功"]},
        {"name": "密码错误", "keywords": ["密码错误"]},
    ]
    rate, covered, missed = run_eval.compute_coverage(cases, scenarios)
    assert rate == 0.5
    assert "成功登录" in covered
    assert "密码错误" in missed


# ----------------------------------------------------------------------
# 5. Mock 模型（Day 6 起）
# ----------------------------------------------------------------------
@pytest.mark.parametrize("style", ["clean", "fenced", "chatty", "broken"])
def test_mock_returns_parseable_json(style):
    mock = MockLLM(style=style)
    out = mock.respond("设计登录用例", system="", hint="json")
    data = json_utils.extract_json(out)  # 不应抛异常
    assert "cases" in data


def test_mock_cycle_rotates_to_broken():
    mock = MockLLM(style="cycle")
    outputs = [mock.respond("x", system="", hint="json") for _ in range(4)]
    # 第 4 次（index 3）应为 broken 风格：字段名被改坏
    assert "expectd" in outputs[3]


# ----------------------------------------------------------------------
# 6. 生成模块 + 覆盖自检（Day 7 / Day 9）
# ----------------------------------------------------------------------
def _clean_client() -> llm_client.LLMClient:
    return llm_client.LLMClient(mock_mode="true", mock_style="clean")


def test_generate_clean_returns_full_rate():
    client = _clean_client()
    result = structured.generate("用户登录功能", client=client)
    assert result.cases
    assert result.structure_rate == 1.0
    assert result.attempts == 1


def test_generator_check_coverage_three_types():
    cases: list[TC] = [
        {"id": "T1", "name": "n", "case_type": "正常", "priority": "P0",
         "steps": ["s"], "expected": "e"},
        {"id": "T2", "name": "n", "case_type": "异常", "priority": "P0",
         "steps": ["s"], "expected": "e"},
        {"id": "T3", "name": "n", "case_type": "边界", "priority": "P0",
         "steps": ["s"], "expected": "e"},
    ]
    report = testcase_generator.TestCaseGenerator.check_coverage(cases)
    assert report.ok
    assert report.missing == []


def test_generator_detects_missing_type():
    cases: list[TC] = [
        {"id": "T1", "name": "n", "case_type": "正常", "priority": "P0",
         "steps": ["s"], "expected": "e"},
    ]
    report = testcase_generator.TestCaseGenerator.check_coverage(cases)
    assert not report.ok
    assert "边界" in report.missing
    assert "异常" in report.missing


def test_generator_generate_covers_all_types():
    client = _clean_client()
    gen = testcase_generator.TestCaseGenerator(client=client)
    suite, report = gen.generate("用户登录功能")
    assert report.ok  # clean mock 三类齐全
    assert len(suite["cases"]) >= 6
