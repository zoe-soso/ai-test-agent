"""
AI 缺陷分析器（Day 24 + Day 25）

Day 24：让 AI 根据"失败病历"分析 问题类型 / 可能原因 / 建议。
Day 25：在 Day 24 的基础上，再给结论加两样东西——
        缺陷分类（UI/接口/数据/定位/环境/代码）和 严重程度（P0~P3）。

为什么分成两天？
    Day 24 先让模型"说人话"，证明它能看懂报错；
    Day 25 再把结论结构化、可量化，这样后面才能做"缺陷分类统计"
    和"失败分析准确率"这类面试指标（Day 29）。

和前面模块一致的几个原则：
    1. 解析模型返回的 JSON 必须做防御（模型可能返回脏 JSON）。
    2. 分类/严重程度是枚举，拿到后必须校验，不在白名单就归到"未知/默认"。
    3. 分析失败不能让整个流水线崩——降级成一条"人工查看"结论即可。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agent import json_utils
from agent.failure_collector import FailureRecord
from config import settings
from prompts import loader
from tools.logger import get_logger

logger = get_logger(__name__)

# Day 25：缺陷分类白名单（模型必须二选一，否则视为未知）
DEFECT_CATEGORIES: tuple[str, ...] = (
    "UI问题", "接口问题", "测试数据问题", "元素定位问题", "环境问题", "代码问题",
)
# Day 25：严重程度白名单
DEFECT_SEVERITIES: tuple[str, ...] = ("P0", "P1", "P2", "P3")

# 把模型可能写出来的各种写法，归一化到白名单
_CATEGORY_ALIASES = {
    "ui": "UI问题", "ui问题": "UI问题", "界面": "UI问题", "页面": "UI问题",
    "接口": "接口问题", "api": "接口问题", "后端": "接口问题",
    "数据": "测试数据问题", "测试数据": "测试数据问题", "data": "测试数据问题",
    "定位": "元素定位问题", "locator": "元素定位问题", "selector": "元素定位问题",
    "环境": "环境问题", "env": "环境问题",
    "代码": "代码问题", "脚本": "代码问题", "test code": "代码问题",
}
_SEVERITY_ALIASES = {
    "p0": "P0", "严重": "P0", "blocker": "P0",
    "p1": "P1", "高": "P1", "high": "P1",
    "p2": "P2", "中": "P2", "medium": "P2",
    "p3": "P3", "低": "P3", "low": "P3",
}


@dataclass
class DefectAnalysis:
    """一条失败用例的 AI 分析结论。"""

    test_name: str
    problem_summary: str = ""
    category: str = "未知"               # 落在 DEFECT_CATEGORIES 里才算已知
    severity: str = "P1"                 # 默认 P1，避免低估
    possible_causes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    raw: str = ""                        # 模型原始返回，便于排查

    @property
    def is_classified(self) -> bool:
        return self.category in DEFECT_CATEGORIES

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "problem_summary": self.problem_summary,
            "category": self.category,
            "severity": self.severity,
            "possible_causes": self.possible_causes,
            "suggestions": self.suggestions,
        }


def _norm_category(value: str) -> str:
    v = (value or "").strip()
    if v in DEFECT_CATEGORIES:
        return v
    return _CATEGORY_ALIASES.get(v.lower(), "未知")


def _norm_severity(value: str) -> str:
    v = (value or "").strip()
    if v in DEFECT_SEVERITIES:
        return v
    return _SEVERITY_ALIASES.get(v.lower(), "P1")


class DefectAnalyzer:
    """把一条失败病历交给 LLM，得到结构化的缺陷分析。"""

    def __init__(self, client: Any, template: str = "analyze_failure") -> None:
        """
        client: 任何带 `chat_messages(messages, mock_hint=...) -> str` 的对象
                （真实 LLMClient 或 MockLLM 都行）。
        """
        self.client = client
        self.template = template

    def analyze(self, failure: FailureRecord) -> DefectAnalysis:
        """分析一条失败用例。失败时降级，不抛异常。"""
        prompt = loader.load(
            self.template,
            test_name=failure.test_name,
            error=failure.error or "（无）",
            # traceback 可能很长，截断到 4000 字符，够模型判断即可
            traceback=(failure.traceback or failure.error)[:4000],
            screenshot=failure.screenshot or "（无截图）",
        )
        try:
            raw = self.client.chat_messages(prompt.to_messages(), mock_hint="analyze")
        except Exception as exc:  # noqa: BLE001 - 模型调用失败不能拖垮流水线
            logger.error("缺陷分析调用失败：%s", exc)
            return DefectAnalysis(
                test_name=failure.test_name,
                problem_summary="分析接口调用失败，请人工查看截图与日志。",
                raw=str(exc),
            )

        return self._parse(raw, failure)

    def _parse(self, raw: str, failure: FailureRecord) -> DefectAnalysis:
        try:
            data = json_utils.extract_json(raw)
        except Exception:  # noqa: BLE001 - 模型返回彻底无法解析时，降级而不是崩
            data = None
        if not data:
            logger.warning("缺陷分析返回不是合法 JSON，降级处理")
            return DefectAnalysis(
                test_name=failure.test_name,
                problem_summary="模型返回无法解析，请人工查看失败信息。",
                raw=raw[:500],
            )

        try:
            return DefectAnalysis(
                test_name=failure.test_name,
                problem_summary=str(data.get("problem_summary", "")).strip(),
                category=_norm_category(str(data.get("category", ""))),
                severity=_norm_severity(str(data.get("severity", "P1"))),
                possible_causes=[str(x) for x in (data.get("possible_causes") or []) if str(x).strip()],
                suggestions=[str(x) for x in (data.get("suggestions") or []) if str(x).strip()],
                raw=raw[:500],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("缺陷分析结构异常：%s", exc)
            return DefectAnalysis(
                test_name=failure.test_name,
                problem_summary="模型返回结构异常，请人工查看。",
                raw=raw[:500],
            )
