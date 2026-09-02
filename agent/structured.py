"""
结构化输出主链路（Day 7 收口）

把 Day 7 的四块拼成一条能用的链路：

    需求
      ↓
    ① prompt 模板（prompts/testcase_v2_json.txt）
      ↓
    ② LLM 调用（agent/llm_client.py）
      ↓
    ③ 脏输出解析（agent/json_utils.py）
      ↓
    ④ 结构校验（agent/validator.py）
      ↓
    ⑤ 不合格就自修正重试（prompts/fix_json.txt）
      ↓
    list[TestCase]  —— 和 Day 2 手写生成器**完全同一份数据契约**

------------------------------------------------------------------
这里最值得讲的一个设计：自修正（Self-Repair）
------------------------------------------------------------------
模型第一次输出不合规时，常见做法是"再生成一次"（盲目重试）。
但盲目重试有个致命问题：**你没告诉它错在哪，它可能用同样的方式再错一遍。**

自修正的做法是——把校验器报出的原话喂回去：

    "#3 TC_LOGIN_008 丢弃：expected 缺失或为空"
    "#8 TC_LOGIN_009 丢弃：case_type「性能测试」不在契约范围内"

这招效果显著，因为错误描述本身就指明了修改方向。
而且它很便宜：一次修复调用的 token 远少于重新生成。

**这也是本项目的第一个"Agent 味道"**：
程序观察自己的执行结果 → 判断不合格 → 基于反馈调整行为 → 再执行。
虽然只有 20 行，但循环已经闭合了。Day 8 会专门讲这件事。

------------------------------------------------------------------
关于重试次数
------------------------------------------------------------------
默认只修 1 次。为什么不是 3 次？
    - 第 1 次修复的成功率最高，边际收益递减得很快
    - 每次重试都是真金白银的 token
    - 修 2 次还不行，基本说明是 prompt 本身的问题，重试没用，
      应该回头改 prompt（这也是评测脚本存在的意义）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent import json_utils, llm_client, validator
from agent.models import TestCase
from agent.validator import CaseIssue
from prompts import loader
from tools.exceptions import LLMResponseError
from tools.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TEMPLATE = "testcase_v2_json"
FIX_TEMPLATE = "fix_json"


@dataclass
class GenerationResult:
    """一次结构化生成的结果（含过程数据，评测要用）。"""

    feature: str
    cases: list[TestCase] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)   # 每次尝试的原始输出
    issues: list[CaseIssue] = field(default_factory=list)  # 最后一次的校验详情
    total_raw_cases: int = 0     # 模型一共给了多少条（含不合格的）
    attempts: int = 0            # 实际调用次数（1 = 一次成功）
    repairs_used: int = 0        # 用了几次自修正

    @property
    def repaired(self) -> bool:
        """是否经过自修正才成功——这是评测里很有意思的一个指标。"""
        return self.repairs_used > 0

    @property
    def structure_rate(self) -> float:
        """结构可用率：合格用例 / 模型给出的总数。

        这是 Day 7 三个核心指标之一。目标 ≥ 95%。
        分母为 0（连 JSON 都没 parse 出来）时算 0 分，不给自己放水。
        """
        if self.total_raw_cases <= 0:
            return 0.0
        return len(self.cases) / self.total_raw_cases

    @property
    def failed_issues(self) -> list[CaseIssue]:
        return [issue for issue in self.issues if not issue.ok]

    def summary(self) -> dict[str, Any]:
        """给评测脚本和命令行用的摘要。"""
        return {
            "功能": self.feature,
            "调用次数": self.attempts,
            "自修正次数": self.repairs_used,
            "模型给出条数": self.total_raw_cases,
            "合格条数": len(self.cases),
            "结构可用率": f"{self.structure_rate:.0%}",
        }


def _parse_and_validate(
    raw: str, feature_key: str
) -> tuple[list[TestCase], list[CaseIssue], list[CaseIssue], int]:
    """解析 + 校验一步到位，返回 (合格, 全部详情, 失败详情, 模型给的条数)。

    把解析失败也包装成 CaseIssue，这样上层不用区分
    "没 parse 出来" 和 "parse 出来但不合规" 两种情况，
    统一按"有错就修"处理，逻辑简单很多。
    """
    try:
        data = json_utils.extract_json(raw)
        raws = json_utils.extract_cases(data)
    except LLMResponseError as exc:
        issue = CaseIssue(index=0, case_id="<解析失败>", errors=[str(exc)])
        return [], [issue], [issue], 0

    passed, all_issues, failed = validator.validate_cases(raws, feature_key=feature_key)
    return passed, all_issues, failed, len(raws)


def generate(
    feature: str,
    description: str = "（无额外描述）",
    *,
    template: str = DEFAULT_TEMPLATE,
    max_repairs: int = 1,
    client: llm_client.LLMClient | None = None,
) -> GenerationResult:
    """从需求生成结构化测试用例。

    参数：
        feature       被测功能名，如 "用户登录功能"
        description   需求描述
        template      用哪个 prompt 模板（Day 10 会换成带设计方法的版本）
        max_repairs   最多自修正几次（默认 1）
        client        可注入客户端，写测试时用

    返回：
        GenerationResult，合格用例在 .cases 里，过程数据在其余字段。
        注意：**失败不抛异常**，而是返回 partial 结果。
        理由：生成 8 条里挂 2 条，剩下 6 条仍然可用，
        全部丢掉太浪费，也让评测脚本没法算分。
    """
    client = client or llm_client.get_client()
    result = GenerationResult(feature=feature)

    # ---- 第一次生成 ----
    prompt = loader.load(template, feature=feature, description=description)
    raw = client.chat_messages(prompt.to_messages(), mock_hint="json")
    result.raw_outputs.append(raw)
    result.attempts = 1

    cases, issues, failed, total = _parse_and_validate(raw, "AUTO")
    result.cases, result.issues, result.total_raw_cases = cases, issues, total

    logger.info(
        "第 1 次生成：%d/%d 条合格（结构可用率 %.0f%%）",
        len(cases), total, result.structure_rate * 100,
    )

    # ---- 自修正重试 ----
    while failed and result.repairs_used < max_repairs:
        result.repairs_used += 1
        result.attempts += 1

        errors_text = validator.describe_issues(failed)
        logger.warning("启动第 %d 次自修正，反馈给模型的问题：\n%s",
                       result.repairs_used, errors_text)

        fix_prompt = loader.load(FIX_TEMPLATE, errors=errors_text, raw=raw)
        raw = client.chat_messages(fix_prompt.to_messages(), mock_hint="json")
        result.raw_outputs.append(raw)

        cases, issues, failed, total = _parse_and_validate(raw, "AUTO")
        result.cases, result.issues = cases, issues
        # 总数取最大的那次，避免修正后条数变少让可用率虚高
        result.total_raw_cases = max(result.total_raw_cases, total)

        logger.info(
            "第 %d 次修正后：%d/%d 条合格（结构可用率 %.0f%%）",
            result.attempts, len(cases), result.total_raw_cases,
            result.structure_rate * 100,
        )

    if result.failed_issues:
        logger.warning(
            "仍有 %d 条不合格，已丢弃：", len(result.failed_issues)
        )
        for issue in result.failed_issues:
            logger.warning("  %s", issue)

    return result


def generate_to_yaml(
    feature: str,
    description: str = "（无额外描述）",
    out_path: str | None = None,
    **kwargs: Any,
) -> tuple[GenerationResult, str]:
    """生成 + 落盘 YAML 一步到位，供命令行调用。

    返回 (生成结果, YAML 文件路径)。
    """
    from pathlib import Path

    from config import settings
    from tools import file_io

    result = generate(feature, description, **kwargs)

    suite = {"feature": feature, "cases": result.cases}
    target = Path(out_path) if out_path else settings.OUTPUT_DIR / "testcases_ai.yaml"
    file_io.write_yaml(target, suite)

    logger.info("已写入 YAML：%s", target)
    return result, str(target)
