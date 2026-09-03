"""
测试用例质量检查器（Day 11）

计划要求：
    让第二次 LLM 调用检查：是否重复？是否覆盖核心场景？是否缺少异常？
    是否存在不合理步骤？

------------------------------------------------------------------
这一天最重要的工程决策：检查分两层，不要什么都问 LLM
------------------------------------------------------------------
新手拿到这个需求，第一反应是"把用例丢给 LLM 让它检查"——全交给 AI。
这在真实工程里是**又贵又不可靠**的做法。原因有两条：

    1. **贵**：每次评审都是一次真实付费调用。用例越多，token 越多。
    2. **不可靠**：LLM 对"数数"和"精确比对"这类任务天生不擅长。
       你问它"这 8 条用例里有重复吗"，它有时明明看到了也说没有。
       但这类问题用几行 Python 就能 100% 确定地检查出来。

所以正确做法是分成两层：

    ┌─ 第一层：规则层（免费、确定性、毫秒级）──────────────┐
    │  重复 ID / 重复用例 / 空步骤 / 预期结果含糊 /         │
    │  类型分布缺失（没有异常或边界）/ 优先级分布不合理      │
    │  → Python 直接算，不需要 LLM，结果可复现              │
    └──────────────────────────────────────────────────┘
                          ↓ 规则查不出的问题
    ┌─ 第二层：LLM 层（花钱、语义、有创意）──────────────┐
    │  是否覆盖了这个功能的核心业务场景？                   │
    │  是否漏掉了重要的异常分支？                          │
    │  步骤描述是否符合真实用户操作逻辑？                   │
    │  → 这类问题没有固定规则，只能靠语义理解，该花这个钱    │
    └──────────────────────────────────────────────────┘

一句话原则：**能用规则解决的，不要问 LLM。**
这条原则后面 Day 17 生成代码、Day 23 分析缺陷时还会反复用到。

------------------------------------------------------------------
为什么这不是"多此一举"
------------------------------------------------------------------
很多人觉得"AI 生成的东西再让 AI 检查"是套娃。其实不是：

    生成时，模型的注意力在"产出内容"上；
    评审时，给它的是"审查清单"，它的注意力在"找问题"上。

这是两个不同的任务，用两次调用分开做，质量明显好于一次调用让它又写又改。
这就是计划里说的"这时候 Agent 的味道开始出来了"——
**生成 → 检查 → 修改**，程序第一次对自己的产出做质量控制。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from agent import json_utils, llm_client, validator
from agent.models import TestCase
from prompts import loader
from tools.logger import get_logger

logger = get_logger(__name__)

REVIEW_TEMPLATE = "review"
REVISE_TEMPLATE = "revise"

# 严重级别。数字用于排序，展示时"错误"排最前面
SEVERITY_ORDER = {"错误": 0, "警告": 1, "建议": 2}

# 预期结果里如果出现这些词（且整体很短），说明写得太含糊。
# 这是测试专业的体现："预期结果：正常" 等于没写，
# 执行测试的人根本没法判断什么叫"正常"。
VAGUE_EXPECTED = {
    "正常", "成功", "通过", "ok", "okay", "失败", "报错", "无", "无异常",
    "正常显示", "正常跳转", "正常提交", "提示错误", "有提示",
}
VAGUE_MAX_LEN = 6  # 预期结果短于这个字数才判定含糊

# 各类问题最少要有几条。这是测试常识：
# 一个功能只测"正常"不测"异常"，上线必出事。
MIN_PER_TYPE = {"正常": 1, "异常": 1, "边界": 1}


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class ReviewIssue:
    """一条评审发现的问题。"""

    severity: str        # 错误 / 警告 / 建议
    category: str        # 重复 / 覆盖 / 异常缺失 / 步骤 / 预期 / 优先级 / 结构
    message: str         # 问题描述（给人看）
    case_id: str = "-"   # 关联的用例 ID；"-" 表示这是整体性问题
    suggestion: str = ""  # 修改建议
    source: str = "规则"  # 规则 / LLM —— 用来统计"多少问题是免费查出来的"

    def __str__(self) -> str:
        head = f"[{self.severity}/{self.category}]"
        if self.case_id != "-":
            head += f" {self.case_id}"
        text = f"{head} {self.message}"
        if self.suggestion:
            text += f" ｜ 建议：{self.suggestion}"
        return text

    @property
    def order(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 9)


@dataclass
class ReviewReport:
    """一次评审的完整结果。"""

    feature: str
    issues: list[ReviewIssue] = field(default_factory=list)
    overall: str = ""                        # LLM 的一句话总评
    missing_scenarios: list[str] = field(default_factory=list)  # LLM 认为漏掉的场景
    llm_calls: int = 0                       # 这次评审花了几次 LLM 调用

    # ---- 分级查询 ----
    def by_severity(self, severity: str) -> list[ReviewIssue]:
        return [i for i in self.issues if i.severity == severity]

    @property
    def blocking(self) -> list[ReviewIssue]:
        """错误级问题——这些必须改，否则用例集不合格。"""
        return self.by_severity("错误")

    @property
    def passed(self) -> bool:
        """是否通过评审（没有错误级问题）。"""
        return not self.blocking

    def sorted_issues(self) -> list[ReviewIssue]:
        return sorted(self.issues, key=lambda i: i.order)

    # ---- 统计 ----
    @property
    def rule_issues(self) -> list[ReviewIssue]:
        return [i for i in self.issues if i.source == "规则"]

    @property
    def llm_issues(self) -> list[ReviewIssue]:
        return [i for i in self.issues if i.source == "LLM"]

    def summary(self) -> dict[str, Any]:
        return {
            "功能": self.feature,
            "问题总数": len(self.issues),
            "错误": len(self.by_severity("错误")),
            "警告": len(self.by_severity("警告")),
            "建议": len(self.by_severity("建议")),
            "规则层查出": len(self.rule_issues),
            "LLM 层查出": len(self.llm_issues),
            "遗漏场景": len(self.missing_scenarios),
            "评审调用": self.llm_calls,
        }

    def describe(self) -> str:
        """给命令行用的一句话结论。"""
        if self.passed:
            tail = "（仅有警告/建议）" if self.issues else ""
            return f"评审通过{tail}"
        return f"未通过：{len(self.blocking)} 个错误级问题"

    def to_text(self) -> str:
        """把问题清单渲染成文本，喂给"修改"环节的 prompt。"""
        if not self.issues:
            return "（规则层未发现问题）"

        lines = []
        for i, issue in enumerate(self.sorted_issues(), 1):
            line = f"{i}. [{issue.severity}] {issue.category}"
            if issue.case_id != "-":
                line += f" 用例 {issue.case_id}"
            line += f"：{issue.message}"
            if issue.suggestion:
                line += f"（建议：{issue.suggestion}）"
            lines.append(line)

        if self.missing_scenarios:
            lines.append("")
            lines.append("遗漏的场景：" + "、".join(self.missing_scenarios))

        return "\n".join(lines)


# ----------------------------------------------------------------------
# 第一层：规则检查（免费、确定性）
# ----------------------------------------------------------------------
_PUNCT = re.compile(
    r"[\s，。、；：！？,.!?;:（）()\[\]【】\"'“”‘’\-—_/\\|]+"
)


def _fingerprint(text: str) -> str:
    """把文字压成"指纹"：去掉标点和空格、统一小写。

    "  密码 错误  "、"密码错误"、"密码，错误" 会得到同一个指纹。
    为什么要这么做？
        模型生成的用例，重复项往往只差一个标点或一个空格。
        直接 == 比较会漏掉这些"实质重复"，而指纹比对能抓到。
    """
    return _PUNCT.sub("", str(text)).lower()


def check_duplicates(cases: list[TestCase]) -> list[ReviewIssue]:
    """查重复：重复的 ID、重复的名称、重复的步骤。

    分三种查，因为"重复"有好几种形态：
        1. ID 撞车    → 用例编号唯一是硬要求，必须改
        2. 名称相同   → 大概率是同一条写了两遍
        3. 名称不同但步骤完全一样 → 更隐蔽，本质还是同一条
    """
    issues: list[ReviewIssue] = []

    # 1) 重复 ID
    id_count: dict[str, int] = {}
    for case in cases:
        cid = str(case.get("id", "")).strip()
        if cid:
            id_count[cid] = id_count.get(cid, 0) + 1
    for cid, count in id_count.items():
        if count > 1:
            issues.append(ReviewIssue(
                severity="错误",
                category="重复",
                message=f"用例编号 {cid} 重复出现 {count} 次",
                case_id=cid,
                suggestion="编号必须唯一，请重新编号",
            ))

    # 2) 名称指纹重复
    name_map: dict[str, list[str]] = {}
    for case in cases:
        key = _fingerprint(case.get("name", ""))
        if key:
            name_map.setdefault(key, []).append(str(case.get("id", "?")))
    for ids in name_map.values():
        if len(ids) > 1:
            issues.append(ReviewIssue(
                severity="错误",
                category="重复",
                message=f"用例 {'、'.join(ids)} 的名称实质相同",
                case_id=ids[0],
                suggestion="合并成一条，或改成真正不同的场景",
            ))

    # 3) 步骤指纹重复（名称不同但做的事一样）
    steps_map: dict[str, list[str]] = {}
    for case in cases:
        key = "".join(_fingerprint(s) for s in case.get("steps", []))
        if key:
            steps_map.setdefault(key, []).append(str(case.get("id", "?")))
    for ids in steps_map.values():
        if len(ids) > 1:
            # 名称已判重过的会重复报，这里只报名称不同、步骤相同的
            names = {_fingerprint(c.get("name", "")) for c in cases
                     if str(c.get("id", "?")) in ids}
            if len(names) > 1:
                issues.append(ReviewIssue(
                    severity="警告",
                    category="重复",
                    message=f"用例 {'、'.join(ids)} 的操作步骤完全一致",
                    case_id=ids[0],
                    suggestion="步骤相同意味着测的是同一件事，建议合并",
                ))

    return issues


def check_case_quality(cases: list[TestCase]) -> list[ReviewIssue]:
    """查单条用例的质量：空步骤、预期结果含糊。"""
    issues: list[ReviewIssue] = []

    for case in cases:
        cid = str(case.get("id", "?"))

        # 空步骤
        steps = case.get("steps") or []
        if not steps:
            issues.append(ReviewIssue(
                severity="错误",
                category="步骤",
                message=f"{cid} 没有操作步骤",
                case_id=cid,
                suggestion="补上可执行的步骤，否则无法执行",
            ))
        elif len(steps) == 1 and len(str(steps[0])) < 4:
            issues.append(ReviewIssue(
                severity="警告",
                category="步骤",
                message=f"{cid} 只有 1 步且过于简略：{steps[0]}",
                case_id=cid,
                suggestion="拆成具体动作，如 打开页面 → 输入账号 → 点击登录",
            ))

        # 预期结果含糊
        expected = _fingerprint(case.get("expected", ""))
        if not expected:
            issues.append(ReviewIssue(
                severity="错误",
                category="预期",
                message=f"{cid} 缺少预期结果",
                case_id=cid,
                suggestion="预期结果是测试用例的灵魂，必须写清楚",
            ))
        elif len(expected) <= VAGUE_MAX_LEN and expected in {
            _fingerprint(v) for v in VAGUE_EXPECTED
        }:
            issues.append(ReviewIssue(
                severity="警告",
                category="预期",
                message=f"{cid} 预期结果太含糊：「{case.get('expected')}」",
                case_id=cid,
                # 建议只给"写法"，不给具体例子。
                # 因为这条用例可能是异常场景，举个"登录成功"的例子会误导修改方向。
                suggestion="写明可验证的结果（具体提示文案或页面状态变化）",
            ))

    return issues


def check_distribution(cases: list[TestCase]) -> list[ReviewIssue]:
    """查整体分布：类型是否齐全、优先级是否合理。

    这是**测试专业度**的集中体现。
    模型很容易生成一堆"正常场景"就交差，
    但真正会出问题的恰恰是异常和边界——所以必须用规则卡住。
    """
    issues: list[ReviewIssue] = []
    if not cases:
        return [ReviewIssue(
            severity="错误", category="结构",
            message="用例集为空", suggestion="至少要有 1 条用例",
        )]

    # 类型分布
    type_count: dict[str, int] = {}
    for case in cases:
        type_count[case.get("case_type", "?")] = type_count.get(case.get("case_type", "?"), 0) + 1

    for need_type, minimum in MIN_PER_TYPE.items():
        actual = type_count.get(need_type, 0)
        if actual < minimum:
            issues.append(ReviewIssue(
                severity="警告",
                category="覆盖" if need_type == "正常" else "异常缺失",
                message=f"缺少「{need_type}」场景（当前 {actual} 条，建议至少 {minimum} 条）",
                suggestion=f"补充 {need_type} 场景用例，这是必测项",
            ))

    # 优先级分布：全 P0 等于没有优先级，全 P3 说明没抓住重点
    priorities = [case.get("priority", "?") for case in cases]
    if priorities and all(p == "P0" for p in priorities):
        issues.append(ReviewIssue(
            severity="建议",
            category="优先级",
            message=f"{len(cases)} 条用例全是 P0",
            suggestion="优先级要区分开，全 P0 等于没排优先级",
        ))
    elif "P0" not in priorities:
        issues.append(ReviewIssue(
            severity="警告",
            category="优先级",
            message="没有 P0 用例",
            suggestion="至少要有一条最高优先级的主流程用例",
        ))

    return issues


def check_by_rules(cases: list[TestCase]) -> list[ReviewIssue]:
    """跑完所有规则检查。免费、确定、毫秒级。"""
    issues: list[ReviewIssue] = []
    issues.extend(check_duplicates(cases))
    issues.extend(check_case_quality(cases))
    issues.extend(check_distribution(cases))
    return issues


# ----------------------------------------------------------------------
# 第二层：LLM 语义评审（花钱，但规则做不到）
# ----------------------------------------------------------------------
def _render_cases(cases: list[TestCase]) -> str:
    """把用例序列化成给模型看的 JSON。

    为什么要精简字段？
        用例里有 design_method 这类元数据，对评审没有帮助，
        带上只是白白消耗 token。评审只需要看"测了什么、怎么测、期望什么"。
    """
    slim = [
        {
            "id": c.get("id"),
            "name": c.get("name"),
            "case_type": c.get("case_type"),
            "priority": c.get("priority"),
            "steps": c.get("steps", []),
            "expected": c.get("expected"),
        }
        for c in cases
    ]
    return json.dumps(slim, ensure_ascii=False, indent=2)


def review_by_llm(
    feature: str,
    cases: list[TestCase],
    client: llm_client.LLMClient | None = None,
) -> ReviewReport:
    """让 LLM 做语义评审，返回填充了 LLM 意见的报告。

    注意：**失败不抛异常**。
        评审是"锦上添花"的环节，LLM 挂了不应该让整条流水线崩掉。
        规则层的结果已经够用了，LLM 挂了就优雅降级。
        这个原则叫"降级不中断"（graceful degradation）。
    """
    client = client or llm_client.get_client()
    report = ReviewReport(feature=feature)

    if not cases:
        report.overall = "（用例集为空，跳过 LLM 评审）"
        return report

    prompt = loader.load(
        REVIEW_TEMPLATE,
        feature=feature,
        cases=_render_cases(cases),
        count=len(cases),
    )

    try:
        # mock_hint="review"：评审要的是"评审结论"结构，不是用例列表。
        # 这里如果传 "json"，离线模式下 mock 会返回用例 JSON，
        # 解析成功了但取不到 issues/overall，评审等于静默失效 —— 必须显式区分。
        raw = client.chat_messages(prompt.to_messages(), mock_hint="review")
        report.llm_calls = 1
    except Exception as exc:  # noqa: BLE001 - 评审失败不应中断主流程
        logger.warning("LLM 评审调用失败，降级为仅规则层结果：%s", exc)
        report.overall = f"（LLM 评审不可用：{type(exc).__name__}）"
        return report

    # 解析模型的评审结论。解析不出来不影响规则层结果
    try:
        data = json_utils.extract_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("评审结果解析失败（%s），原始返回：%s", exc, raw[:200])
        report.overall = "（评审返回无法解析，仅保留规则层结果）"
        return report

    if not isinstance(data, dict):
        report.overall = "（评审返回格式异常，仅保留规则层结果）"
        return report

    report.overall = str(data.get("overall", "") or "")

    missing = data.get("missing_scenarios") or []
    if isinstance(missing, list):
        report.missing_scenarios = [str(m) for m in missing if str(m).strip()]

    raw_issues = data.get("issues") or []
    if isinstance(raw_issues, list):
        for item in raw_issues:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity", "建议")).strip()
            if severity not in SEVERITY_ORDER:
                severity = "建议"  # 模型乱写级别就降级为建议，不让它制造假错误
            report.issues.append(ReviewIssue(
                severity=severity,
                category=str(item.get("category", "其他")).strip() or "其他",
                message=str(item.get("problem", "")).strip() or "（未说明问题）",
                case_id=str(item.get("case_id", "-")).strip() or "-",
                suggestion=str(item.get("suggestion", "")).strip(),
                source="LLM",
            ))

    return report


# ----------------------------------------------------------------------
# 对外主入口
# ----------------------------------------------------------------------
def review(
    feature: str,
    cases: list[TestCase],
    *,
    client: llm_client.LLMClient | None = None,
    use_llm: bool = True,
) -> ReviewReport:
    """评审一组测试用例。

    参数：
        feature    被测功能名
        cases      待评审的用例
        client     可注入的客户端（测试用）
        use_llm    是否启用 LLM 层。关掉就是纯规则检查，零成本零延迟。
    """
    # 规则层永远先跑：免费、确定、能挡掉大部分低级问题
    rule_issues = check_by_rules(cases)

    if use_llm:
        report = review_by_llm(feature, cases, client=client)
    else:
        report = ReviewReport(feature=feature)

    # 合并：规则层在前，因为它是"确定事实"，比模型的判断更硬
    report.issues = rule_issues + report.issues

    logger.info(
        "评审完成：%s（规则层 %d 条，LLM 层 %d 条）",
        report.describe(), len(report.rule_issues), len(report.llm_issues),
    )
    return report


# ----------------------------------------------------------------------
# 修改：根据评审意见让 LLM 改用例
# ----------------------------------------------------------------------
def revise(
    feature: str,
    cases: list[TestCase],
    report: ReviewReport,
    *,
    client: llm_client.LLMClient | None = None,
) -> tuple[list[TestCase], bool]:
    """根据评审意见修改用例。

    返回 (修改后的用例, 是否真的改成功了)。

    ------------------------------------------------------------------
    这里有一条铁律：修改失败，必须退回原用例
    ------------------------------------------------------------------
    用例是"资产"，已经生成出来了，不能因为一次修改失败就全丢。
    所以任何环节出问题（调用失败 / 解析失败 / 修改后反而变少），
    都返回原来的 cases，让上层继续用。

    这看起来是常识，但工程里最常见的 bug 恰恰是：
    用一个可能失败的操作去覆盖已有的好数据。
    """
    client = client or llm_client.get_client()

    prompt = loader.load(
        REVISE_TEMPLATE,
        feature=feature,
        cases=_render_cases(cases),
        issues=report.to_text(),
        count=len(cases),
    )

    try:
        raw = client.chat_messages(prompt.to_messages(), mock_hint="json")
    except Exception as exc:  # noqa: BLE001
        logger.warning("修改用例时 LLM 调用失败，保留原用例：%s", exc)
        return cases, False

    try:
        data = json_utils.extract_json(raw)
        raws = json_utils.extract_cases(data)
        passed, _, failed = validator.validate_cases(raws, feature_key="AUTO")
    except Exception as exc:  # noqa: BLE001
        logger.warning("修改结果解析失败，保留原用例：%s", exc)
        return cases, False

    if not passed:
        logger.warning("修改后没有一条合格，保留原用例")
        return cases, False

    # 修改后条数明显变少（比如原来 8 条改完只剩 3 条），
    # 大概率是模型"偷懒"把用例删了而不是改了 —— 这种情况也退回
    if len(passed) < len(cases) * 0.6:
        logger.warning(
            "修改后用例从 %d 条降到 %d 条（降幅过大），判定为模型误删，保留原用例",
            len(cases), len(passed),
        )
        return cases, False

    if failed:
        logger.info("修改结果中丢弃了 %d 条不合格用例", len(failed))

    return passed, True


def review_and_revise(
    feature: str,
    cases: list[TestCase],
    *,
    client: llm_client.LLMClient | None = None,
    max_rounds: int = 1,
    use_llm: bool = True,
) -> tuple[list[TestCase], ReviewReport, int]:
    """生成 → Review → 修改 的完整闭环。

    返回 (最终用例, 最终评审报告, 实际修改轮数)。

    max_rounds 默认是 1，不是 3。为什么？
        - 第 1 轮修复的收益最大，之后急剧递减
        - 每轮都是真金白银
        - 改 1 轮还有错误级问题，说明是 prompt 或需求本身有问题，
          改再多轮也没用，应该回头改生成阶段的 prompt
        这个判断和 Day 7 自修正"最多修 1 次"是同一个道理。
    """
    client = client or llm_client.get_client()
    rounds = 0

    report = review(feature, cases, client=client, use_llm=use_llm)
    current = cases

    while report.blocking and rounds < max_rounds:
        rounds += 1
        logger.info(
            "第 %d 轮修改：有 %d 个错误级问题需要处理", rounds, len(report.blocking)
        )

        revised, ok = revise(feature, current, report, client=client)
        if not ok:
            logger.warning("第 %d 轮修改未成功，停止并保留当前用例", rounds)
            break

        current = revised
        # 重新评审，看改完还有没有问题
        report = review(feature, current, client=client, use_llm=use_llm)

    return current, report, rounds
