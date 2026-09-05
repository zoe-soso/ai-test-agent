"""
Day 29 评测脚本的冒烟测试（全部离线 Mock，不联网、不启动浏览器）。

只验证三件事：
  1. 三个评测脚本都能跑完、返回结构合理的结果（不崩）
  2. 算出来的比率都在 [0, 1] 区间
  3. 样本数符合预期

真实的"准确率数字"需要配好 API Key 后用真实模型跑才有意义；
Mock 下这些脚本主要验证"评测链路本身能跑通"。
"""

import pytest

from agent import llm_client
from agent import project_profile
from config import settings
from eval import eval_coverage, eval_code_passrate, eval_defect_accuracy


@pytest.fixture
def mock_client():
    return llm_client.LLMClient(mock_mode="true")


def test_defect_accuracy_runs_offline(mock_client):
    labels = eval_defect_accuracy.load_labels(eval_defect_accuracy.DEFAULT_LABELS)
    result = eval_defect_accuracy.evaluate(mock_client, labels)
    assert result.total == len(labels)
    assert 0.0 <= result.category_accuracy <= 1.0
    assert 0.0 <= result.severity_accuracy <= 1.0
    assert 0.0 <= result.classify_rate <= 1.0
    assert len(result.rows) == len(labels)


def test_code_passrate_runs_offline(mock_client):
    profile = project_profile.load_profile("ecommerce")
    result = eval_code_passrate.evaluate(mock_client, profile=profile, max_repairs=1)
    # 样本是登录 / 购物车 / 注册 三条
    assert result.total == 3
    assert 0.0 <= result.first_pass_rate <= 1.0
    assert 0.0 <= result.repaired_pass_rate <= 1.0
    assert result.avg_repairs >= 0
    assert len(result.rows) == 3


def test_coverage_runs_offline(mock_client):
    result = eval_coverage.evaluate(mock_client)
    assert result.total >= 1               # 至少读到一个评测集用例
    assert 0.0 <= result.avg_structure_rate <= 1.0
    assert 0.0 <= result.avg_coverage_rate <= 1.0
    # 每条都应该带功能名
    assert all(r.feature for r in result.rows)
