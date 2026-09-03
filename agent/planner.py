"""
测试用例生成 Agent（Day 14 —— 第一个可演示的 Agent MVP）

计划 Day 14 要的东西：
    用户输入需求 → Agent → 生成测试用例 → Review → 输出 YAML

前面 10 天我们已经有了三块积木：
    - Day 9  TestCaseGenerator   会生成用例，并保证 正常/异常/边界 三类都覆盖
    - Day 11 TestCaseReviewer    会检查用例质量（规则层 + LLM 层）
    - Day 12 TestDataGenerator   会生成测试数据（Day 14 暂不用，留给后面）

今天就是把它们**串起来**变成一条自动流水线：

    ┌──────────────────────────────────────────────────────────┐
    │  需求（一句话 / 一段文字）                                  │
    │       │                                                    │
    │       ▼                                                    │
    │  [1] TestCaseGenerator.generate()   生成用例 + 覆盖自检    │
    │       │                                                    │
    │       ▼                                                    │
    │  [2] TestCaseReviewer.review_and_revise()  质量把关 + 修改 │
    │       │                                                    │
    │       ▼                                                    │
    │  [3] 落盘 YAML（用例 + 评审报告）                          │
    └──────────────────────────────────────────────────────────┘

------------------------------------------------------------------
为什么这叫 "Agent" 而不是 "Chain"（普通链式调用）
------------------------------------------------------------------
关键在 [2] 这一步：程序**先观察（Review）再决策（要不要改、改什么）**，
而不是一条道走到黑。

    观察：这一批用例里有没有重复？异常场景够不够？
    决策：有错误级问题 → 让 LLM 按意见改一轮
    行动：改完再 Review 一遍确认

这就是 Day 8 反复强调的"观察 → 决策 → 行动"闭环。
虽然它还很简单（只有 1 轮修改），但**味道已经对了**——
它对自己的产出做了质量控制，而不是生成完就直接交给你。

------------------------------------------------------------------
给面试讲的一句话
------------------------------------------------------------------
"我的 Agent MVP 能做到：给一句需求，自动生成结构化测试用例，
自己先做质量评审，有问题就按评审意见修改一轮，最后把用例和评审报告
一起落盘。整个生成 → 评审 → 修改的闭环不需要人介入，
人只在最后确认结果。"

（注意：Day 19 会在这里加 Human-in-the-loop——
生成的代码要人点确认才执行；Day 14 只是用例和报告，不涉及执行，
所以先全自动跑通。）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import llm_client, reviewer
from agent.testcase_generator import TestCaseGenerator
from agent.models import TestSuite
from config import settings
from tools import file_io
from tools.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """一次 Agent 运行的完整结果（方便命令行打印，也方便测试断言）。"""

    feature: str
    suite: TestSuite
    review: Any                       # ReviewReport
    revise_rounds: int = 0
    cases_path: str = ""
    review_path: str = ""
    usage: Any = None                 # TokenUsage

    # ---- 给命令行用的摘要 ----
    def summary(self) -> dict[str, Any]:
        cases = self.suite.get("cases", [])
        by_type: dict[str, int] = {}
        for case in cases:
            by_type[case.get("case_type", "?")] = by_type.get(case.get("case_type", "?"), 0) + 1
        return {
            "功能": self.feature,
            "用例总数": len(cases),
            "正常": by_type.get("正常", 0),
            "异常": by_type.get("异常", 0),
            "边界": by_type.get("边界", 0),
            "评审结论": self.review.describe(),
            "修改轮数": self.revise_rounds,
            "用例文件": self.cases_path,
            "评审报告": self.review_path,
        }


class TestCaseAgent:
    """测试用例生成 Agent（MVP）。

    用法：
        agent = TestCaseAgent()
        result = agent.run("用户登录功能")
        print(result.summary())
    """

    def __init__(
        self,
        client: llm_client.LLMClient | None = None,
        *,
        template: str | None = None,
        auto_fix: bool = True,
        use_llm: bool = True,
        max_repairs: int = 1,
    ) -> None:
        self.client = client or llm_client.get_client()
        self.template = template
        self.auto_fix = auto_fix
        self.use_llm = use_llm
        self.max_repairs = max_repairs

    # ------------------------------------------------------------------
    def run(
        self,
        feature: str,
        description: str = "（无额外描述）",
        *,
        cases_path: str | Path | None = None,
        review_path: str | Path | None = None,
    ) -> AgentResult:
        """跑完"生成 → 评审 → 落盘"整条链路。

        参数都给了默认值，最常用的一行就能跑：
            agent.run("用户登录功能")
        """
        # ---- 1. 生成 ----
        generator = TestCaseGenerator(
            client=self.client,
            template=self.template,
            max_repairs=self.max_repairs,
        )
        suite, cov_report = generator.generate(feature, description)
        logger.info("生成阶段完成：%s", cov_report.describe())

        # ---- 2. 评审 + （可选）修改 ----
        if self.auto_fix and self.use_llm:
            final_cases, review, rounds = reviewer.review_and_revise(
                feature, suite["cases"],
                client=self.client, max_rounds=self.max_repairs, use_llm=True,
            )
        else:
            review = reviewer.review(
                feature, suite["cases"], client=self.client, use_llm=self.use_llm
            )
            final_cases, rounds = suite["cases"], 0

        final_suite: TestSuite = {"feature": feature, "cases": final_cases}

        # ---- 3. 落盘 ----
        cases_file = Path(cases_path) if cases_path else settings.OUTPUT_DIR / "testcases_ai.yaml"
        file_io.write_yaml(cases_file, final_suite)
        logger.info("用例已写入：%s（%d 条）", cases_file, len(final_cases))

        review_file = Path(review_path) if review_path else settings.OUTPUT_DIR / "review_report.yaml"
        self._save_review(review_file, feature, review)

        return AgentResult(
            feature=feature,
            suite=final_suite,
            review=review,
            revise_rounds=rounds,
            cases_path=str(cases_file),
            review_path=str(review_file),
            usage=self.client.usage,
        )

    @staticmethod
    def _save_review(path: Path, feature: str, review: Any) -> None:
        """把评审报告存成 YAML，方便日后回顾 / 做评测基线。"""
        payload = {
            "feature": feature,
            "结论": review.describe(),
            "问题总数": len(review.issues),
            "错误": len(review.by_severity("错误")),
            "警告": len(review.by_severity("警告")),
            "建议": len(review.by_severity("建议")),
            "规则层查出": len(review.rule_issues),
            "LLM 层查出": len(review.llm_issues),
            "遗漏场景": review.missing_scenarios,
            "总评": review.overall,
            "问题清单": [
                {
                    "严重级别": i.severity,
                    "类别": i.category,
                    "用例": i.case_id,
                    "描述": i.message,
                    "建议": i.suggestion,
                    "来源": i.source,
                }
                for i in review.sorted_issues()
            ],
        }
        file_io.write_yaml(path, payload)
        logger.info("评审报告已写入：%s", path)
