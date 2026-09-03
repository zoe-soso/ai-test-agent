"""
用例结构校验与规范化（Day 7 第二块）

------------------------------------------------------------------
这个文件解决什么问题
------------------------------------------------------------------
模型返回的 JSON 能 parse 成功，只是过了第一关。
它依然可能：

    - 少给你 expected 字段（没预期结果 = 没法转成断言 = 这条废了）
    - case_type 写成 "性能测试"（契约里只有 正常/异常/边界）
    - priority 写成 "P9"（契约里只有 P0~P3）
    - steps 给了一整段字符串，而不是步骤列表
    - 同一次生成里两条用例 id 重复

**能 parse ≠ 能用。** 这一步就是把"能 parse"变成"能用"，
并且**把不合规的部分明确统计出来**——这些统计就是 Day 7 评测指标
「结构可用率」的数据来源。

------------------------------------------------------------------
策略：能救的救，救不了的丢
------------------------------------------------------------------
分两类问题，处理方式完全不同：

    可修复（warning）—— 语义明确，只是写法不标准
        "正向"        -> "正常"
        "高"          -> "P0"
        "P9"          -> "P3"（数字越界，钳到最近的合法值）
        steps 是字符串 -> 按行/箭头拆成列表
        id 缺失        -> 自动生成 TC_AUTO_001

    致命错误（error）—— 信息缺失或超出契约，只能丢弃
        缺 expected            -> 没有预期结果，无法断言
        case_type 无法归类     -> "性能测试"这种，说明模型在自由发挥
        steps 为空             -> 没有步骤，自动化无从下手
        name 为空              -> 连名字都没有

为什么要丢弃而不是硬凑？
    宁可少两条用例，也不要把一条"看起来像用例"的垃圾
    混进最终产出里——它会在 Day 17 生成 Playwright 代码时炸掉，
    而且那时候你已经分不清是模型的锅还是你的锅。

    丢弃的比例会被记录下来。如果某天这个比例突然升高，
    说明你改的 prompt 有问题，或者模型抽风了——这是重要的信号。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent.models import (
    DATA_TYPES,
    DESIGN_METHODS,
    CaseType,
    DataType,
    DesignMethod,
    Priority,
    TestCase,
    TestData,
)
from tools.logger import get_logger

logger = get_logger(__name__)

VALID_CASE_TYPES: tuple[CaseType, ...] = ("正常", "异常", "边界")
VALID_PRIORITIES: tuple[Priority, ...] = ("P0", "P1", "P2", "P3")

# 模型爱用的同义词。语义明确，可以安全地映射过去。
CASE_TYPE_ALIASES: dict[str, CaseType] = {
    "正常": "正常", "正向": "正常", "正常场景": "正常", "正向场景": "正常",
    "happy": "正常", "positive": "正常", "有效": "正常", "有效等价类": "正常",
    "异常": "异常", "异常场景": "异常", "反向": "异常", "无效": "异常",
    "negative": "异常", "错误": "异常", "失败": "异常",
    "边界": "边界", "边界值": "边界", "边界场景": "边界", "临界": "边界",
    "boundary": "边界", "edge": "边界", "极限": "边界",
}

PRIORITY_ALIASES: dict[str, Priority] = {
    "p0": "P0", "高": "P0", "最高": "P0", "high": "P0", "critical": "P0", "紧急": "P0",
    "p1": "P1", "中": "P1", "medium": "P1", "较高": "P1", "重要": "P1",
    "p2": "P2", "低": "P2", "low": "P2", "较低": "P2", "一般": "P2",
    "p3": "P3", "最低": "P3", "lowest": "P3", "边缘": "P3", "轻微": "P3",
}


@dataclass
class CaseIssue:
    """一条用例的校验结果（不管通过与否都有）。"""

    index: int
    case_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """没有致命错误就算通过（warning 不算失败）。"""
        return not self.errors

    def __str__(self) -> str:
        if self.ok:
            extra = f"，{len(self.warnings)} 处已自动修复" if self.warnings else ""
            return f"#{self.index} {self.case_id} 通过{extra}"
        return f"#{self.index} {self.case_id} 丢弃：{'; '.join(self.errors)}"


# ----------------------------------------------------------------------
# 字段级规范化
# ----------------------------------------------------------------------
def _norm_steps(value: Any, issue: CaseIssue) -> list[str]:
    """把 steps 规范成 list[str]。

    模型可能给：
        ["打开首页", "点击登录"]              <- 正常
        "打开首页 -> 点击登录 -> 输入账号"     <- 一整段
        "1. 打开首页\\n2. 点击登录"           <- 带编号的一整段
    """
    if isinstance(value, list):
        steps = [str(item).strip() for item in value if str(item).strip()]
        if not steps:
            issue.errors.append("steps 是空列表")
        return steps

    if isinstance(value, str):
        text = value.strip()
        if not text:
            issue.errors.append("steps 是空字符串")
            return []

        # 先按换行切，再按箭头切
        parts: list[str] = []
        for line in text.splitlines():
            for chunk in re.split(r"[→\->]|->", line):
                chunk = chunk.strip()
                if chunk:
                    parts.append(chunk)

        # 去掉行首的 "1." "2)" 这类编号
        cleaned = [re.sub(r"^\d+\s*[.、)．]\s*", "", p).strip() for p in parts]
        cleaned = [c for c in cleaned if c]

        if cleaned:
            issue.warnings.append(f"steps 原本是字符串，已拆成 {len(cleaned)} 步")
            return cleaned

        issue.errors.append("steps 无法拆出有效步骤")
        return []

    issue.errors.append(f"steps 类型错误：{type(value).__name__}")
    return []


def _norm_case_type(value: Any, issue: CaseIssue) -> CaseType | None:
    """把 case_type 规范成 正常/异常/边界 之一。"""
    if not isinstance(value, str) or not value.strip():
        issue.errors.append("case_type 缺失或不是字符串")
        return None

    raw = value.strip()
    if raw in VALID_CASE_TYPES:
        return raw  # type: ignore[return-value]

    hit = CASE_TYPE_ALIASES.get(raw) or CASE_TYPE_ALIASES.get(raw.lower())
    if hit:
        issue.warnings.append(f"case_type「{raw}」已映射为「{hit}」")
        return hit

    issue.errors.append(
        f"case_type「{raw}」不在契约范围内（只能是 正常/异常/边界）"
    )
    return None


def _norm_priority(value: Any, issue: CaseIssue) -> Priority:
    """把 priority 规范成 P0~P3 之一。

    优先级即使写错也不至于让这条用例作废，
    所以最差情况兜底成 P3，只记 warning，不丢弃。
    """
    if not isinstance(value, str) or not value.strip():
        issue.warnings.append("priority 缺失，兜底为 P2")
        return "P2"

    raw = value.strip()
    upper = raw.upper()
    if upper in VALID_PRIORITIES:
        return upper  # type: ignore[return-value]

    hit = PRIORITY_ALIASES.get(upper) or PRIORITY_ALIASES.get(raw)
    if hit:
        issue.warnings.append(f"priority「{raw}」已映射为「{hit}」")
        return hit

    # 形如 P5 / P9：数字越界但有明确方向，钳到最近的合法值
    matched = re.fullmatch(r"[Pp](\d+)", raw)
    if matched:
        level = min(int(matched.group(1)), 3)
        issue.warnings.append(f"priority「{raw}」越界，已钳为「P{level}」")
        return f"P{level}"  # type: ignore[return-value]

    issue.warnings.append(f"priority「{raw}」无法识别，兜底为 P2")
    return "P2"


# 模型对设计方法的叫法五花八门，语义明确的都映射过来。
# 注意：识别不了就记"未标注"，**不因此丢弃这条用例** ——
# design_method 只是元数据，用例本身可能是好的，不能因噎废食。
DESIGN_METHOD_ALIASES: dict[str, DesignMethod] = {
    "等价类": "等价类划分", "等价类划分": "等价类划分", "等价类划分法": "等价类划分",
    "有效等价类": "等价类划分", "无效等价类": "等价类划分", "equivalence": "等价类划分",
    "边界值": "边界值分析", "边界值分析": "边界值分析", "边界值分析法": "边界值分析",
    "边界": "边界值分析", "临界值": "边界值分析", "boundary": "边界值分析",
    "判定表": "判定表", "判定表法": "判定表", "判定表驱动": "判定表",
    "decision table": "判定表", "条件组合": "判定表",
    "场景法": "场景法", "场景分析": "场景法", "业务流程": "场景法",
    "业务流": "场景法", "scenario": "场景法", "流程分析": "场景法",
    "异常测试": "异常测试", "异常分析": "异常测试", "异常法": "异常测试",
    "错误推测": "异常测试", "错误猜测": "异常测试", "错误推测法": "异常测试",
}


def _norm_design_method(value: Any, issue: CaseIssue) -> DesignMethod:
    """把 design_method 规范成契约里的五种方法之一。

    认不出来就返回"未标注"，只记 warning，不影响用例本身是否合格。
    """
    if not isinstance(value, str) or not value.strip():
        issue.warnings.append("design_method 未标注")
        return "未标注"

    raw = value.strip()
    if raw in DESIGN_METHODS:
        return raw  # type: ignore[return-value]

    hit = DESIGN_METHOD_ALIASES.get(raw) or DESIGN_METHOD_ALIASES.get(raw.lower())
    if hit:
        issue.warnings.append(f"design_method「{raw}」已映射为「{hit}」")
        return hit

    issue.warnings.append(f"design_method「{raw}」无法识别，记为未标注")
    return "未标注"


# ----------------------------------------------------------------------
# 用例级校验
# ----------------------------------------------------------------------
def validate_case(raw: dict[str, Any], index: int = 0, feature_key: str = "AUTO") -> tuple[TestCase | None, CaseIssue]:
    """校验并规范化一条用例。

    返回：
        (规范化后的 TestCase, 校验详情)
    不合规时第一条是 None，原因写在 CaseIssue.errors 里。

    注意：即使返回 None，CaseIssue 也会带上有用的诊断信息，
    这些会被拿去喂给"自修正"重试（Day 7 第四块）——
    把错误原话告诉模型，它往往能改对。
    """
    issue = CaseIssue(index=index, case_id=str(raw.get("id", f"<无id#{index}>")))

    # ---- id ----
    case_id = str(raw.get("id") or "").strip()
    if not case_id:
        case_id = f"TC_{feature_key.upper()}_{index + 1:03d}"
        issue.warnings.append(f"id 缺失，已自动生成 {case_id}")
    issue.case_id = case_id

    # ---- name ----
    name = str(raw.get("name") or "").strip()
    if not name:
        issue.errors.append("name 缺失或为空")

    # ---- expected（最关键，没有它就没法断言）----
    expected = str(raw.get("expected") or "").strip()
    if not expected:
        # 模型偶尔把 expected 拼成 expectd / expect，认一下
        for typo in ("expectd", "expect", "expect_result", "expected_result"):
            if typo in raw:
                expected = str(raw[typo]).strip()
                if expected:
                    issue.warnings.append(f"expected 字段名为「{typo}」，已兼容读取")
                break
    if not expected:
        issue.errors.append("expected 缺失或为空（没有预期结果就无法转成断言）")

    # ---- case_type ----
    case_type = _norm_case_type(raw.get("case_type"), issue)

    # ---- priority ----
    priority = _norm_priority(raw.get("priority"), issue)

    # ---- steps ----
    steps = _norm_steps(raw.get("steps"), issue)

    # ---- design_method（Day 10 新增，仅元数据，不影响是否合格）----
    design_method = _norm_design_method(raw.get("design_method"), issue)

    if issue.errors or case_type is None:
        if case_type is None and not issue.errors:
            issue.errors.append("case_type 无法归类")
        logger.warning("用例校验未通过 -> %s", issue)
        return None, issue

    case: TestCase = {
        "id": case_id,
        "name": name,
        "case_type": case_type,
        "priority": priority,
        "steps": steps,
        "expected": expected,
        "design_method": design_method,
    }

    if issue.warnings:
        logger.info("用例校验通过（有自动修复）-> %s", issue)
    return case, issue


def validate_cases(
    raws: list[dict[str, Any]],
    feature_key: str = "AUTO",
) -> tuple[list[TestCase], list[CaseIssue], list[CaseIssue]]:
    """批量校验。

    返回：
        (合格用例列表, 全部校验详情, 未通过的详情)

    顺带做两件整体层面的事：
        1. id 去重——重复 id 会让后面的 YAML 和报告对不上号
        2. 统计通过率，这是评测指标的直接数据来源
    """
    passed: list[TestCase] = []
    all_issues: list[CaseIssue] = []
    failed: list[CaseIssue] = []
    seen: dict[str, int] = {}

    for index, raw in enumerate(raws):
        case, issue = validate_case(raw, index=index, feature_key=feature_key)

        # id 去重
        if case is not None:
            if case["id"] in seen:
                seen[case["id"]] += 1
                new_id = f"{case['id']}_DUP{seen[case['id']]}"
                issue.warnings.append(f"id 重复，已重命名为 {new_id}")
                case["id"] = new_id
            else:
                seen[case["id"]] = 1
            passed.append(case)

        all_issues.append(issue)
        if not issue.ok:
            failed.append(issue)

    logger.info(
        "批量校验：共 %d 条，通过 %d 条，丢弃 %d 条",
        len(raws), len(passed), len(failed),
    )
    return passed, all_issues, failed


# ----------------------------------------------------------------------
# Day 12：测试数据校验
# ----------------------------------------------------------------------
# 测试数据的校验比用例**宽松**，这是刻意的：
#
#   用例里 expected 缺失 = 致命（没法断言，这条废了）
#   数据里 expected 缺失 = 只是少个说明，数据本身照样能用
#
# 所以对测试数据，只有"连值都没有"（fields 缺失或为空）才判致命。
# 其他一律降级 + 记录，不丢弃 —— 因为数据生成的成本更高，
# 而且少一组数据往往意味着少覆盖一种场景。
DATA_TYPE_ALIASES: dict[str, DataType] = {
    # 正确数据
    "正确数据": "正确数据", "正确": "正确数据", "正常数据": "正确数据",
    "正常": "正确数据", "有效数据": "正确数据", "有效": "正确数据",
    "合法": "正确数据", "positive": "正确数据", "valid": "正确数据",
    # 错误数据
    "错误数据": "错误数据", "错误": "错误数据", "错误密码": "错误数据",
    "错误值": "错误数据", "无效数据": "错误数据", "无效": "错误数据",
    "negative": "错误数据", "invalid": "错误数据", "wrong": "错误数据",
    # 空值
    "空值": "空值", "空": "空值", "空密码": "空值", "空数据": "空值",
    "留空": "空值", "为空": "空值", "empty": "空值", "null": "空值", "none": "空值",
    # 超长数据
    "超长数据": "超长数据", "超长": "超长数据", "超长密码": "超长数据",
    "超长值": "超长数据", "过长": "超长数据", "long": "超长数据", "overflow": "超长数据",
    # 特殊字符
    "特殊字符": "特殊字符", "特殊字符数据": "特殊字符", "特殊符号": "特殊字符",
    "特殊": "特殊字符", "注入": "特殊字符", "sql注入": "特殊字符",
    "special": "特殊字符", "injection": "特殊字符",
    # 不存在数据
    "不存在数据": "不存在数据", "不存在": "不存在数据", "不存在账号": "不存在数据",
    "未注册": "不存在数据", "未注册账号": "不存在数据", "nonexistent": "不存在数据",
}


def _norm_data_type(value: Any, issue: CaseIssue) -> DataType:
    """把 data_type 规范成契约里的六种之一。

    认不出来记为"未分类"，**不因此丢弃这组数据** ——
    数据类型的归类只是方便统计覆盖率，数据本身仍然可用。
    """
    if not isinstance(value, str) or not value.strip():
        issue.warnings.append("data_type 缺失，记为未分类")
        return "未分类"

    raw = value.strip()
    if raw in DATA_TYPES:
        return raw  # type: ignore[return-value]

    hit = DATA_TYPE_ALIASES.get(raw) or DATA_TYPE_ALIASES.get(raw.lower())
    if hit:
        issue.warnings.append(f"data_type「{raw}」已映射为「{hit}」")
        return hit

    issue.warnings.append(f"data_type「{raw}」无法识别，记为未分类")
    return "未分类"


def _norm_fields(value: Any, expected_params: list[str] | None, issue: CaseIssue) -> dict[str, Any]:
    """校验 fields：必须是非空字典。

    参数名对不上只记 warning —— 模型可能用了更贴切的字段名
    （比如把 username 写成 email），数据本身仍然有效，不该因此丢弃。
    """
    if not isinstance(value, dict):
        issue.errors.append(f"fields 类型错误：{type(value).__name__}（应为对象）")
        return {}

    fields = {str(k): v for k, v in value.items() if str(k).strip()}
    if not fields:
        issue.errors.append("fields 为空（没有实际数据就没法执行）")
        return {}

    if expected_params:
        missing = [p for p in expected_params if p not in fields]
        extra = [k for k in fields if k not in expected_params]
        if missing:
            issue.warnings.append(f"缺少参数：{'、'.join(missing)}")
        if extra:
            issue.warnings.append(f"多余参数：{'、'.join(extra)}")

    return fields


def validate_data_item(
    raw: dict[str, Any],
    index: int = 0,
    params: list[str] | None = None,
    feature_key: str = "AUTO",
) -> tuple[TestData | None, CaseIssue]:
    """校验并规范化一组测试数据。

    返回 (规范化后的 TestData, 校验详情)，不合规时第一条为 None。
    """
    issue = CaseIssue(index=index, case_id=str(raw.get("id", f"<无id#{index}>")))

    # ---- id ----
    data_id = str(raw.get("id") or "").strip()
    if not data_id:
        data_id = f"TD_{feature_key.upper()}_{index + 1:03d}"
        issue.warnings.append(f"id 缺失，已自动生成 {data_id}")
    issue.case_id = data_id

    # ---- name ----
    name = str(raw.get("name") or "").strip()
    if not name:
        # 数据没名字不算致命，用 data_type 凑一个，人还能看懂
        name = f"数据组{index + 1:02d}"
        issue.warnings.append("name 缺失，已用序号代替")

    # ---- fields（唯一的致命项）----
    fields = _norm_fields(raw.get("fields"), params, issue)

    # ---- data_type（元数据，降级不丢弃）----
    data_type = _norm_data_type(raw.get("data_type"), issue)

    # ---- purpose / expected（可选）----
    purpose = str(raw.get("purpose") or "").strip()
    if not purpose:
        purpose = f"验证{data_type}场景"
        issue.warnings.append("purpose 缺失，已按数据类型推断")

    if issue.errors:
        logger.warning("测试数据校验未通过 -> %s", issue)
        return None, issue

    item: TestData = {
        "id": data_id,
        "name": name,
        "data_type": data_type,
        "fields": fields,
        "purpose": purpose,
    }

    expected = str(raw.get("expected") or "").strip()
    if expected:
        item["expected"] = expected

    link_case = str(raw.get("link_case") or "").strip()
    if link_case:
        item["link_case"] = link_case

    return item, issue


def validate_data(
    raws: list[dict[str, Any]],
    params: list[str] | None = None,
    feature_key: str = "AUTO",
) -> tuple[list[TestData], list[CaseIssue], list[CaseIssue]]:
    """批量校验测试数据，返回 (合格数据, 全部详情, 未通过详情)。"""
    passed: list[TestData] = []
    all_issues: list[CaseIssue] = []
    failed: list[CaseIssue] = []
    seen: dict[str, int] = {}

    for index, raw in enumerate(raws):
        if not isinstance(raw, dict):
            issue = CaseIssue(index=index, case_id=f"<非对象#{index}>",
                              errors=[f"数据类型错误：{type(raw).__name__}"])
            all_issues.append(issue)
            failed.append(issue)
            continue

        item, issue = validate_data_item(raw, index=index, params=params,
                                         feature_key=feature_key)
        if item is not None:
            if item["id"] in seen:
                seen[item["id"]] += 1
                new_id = f"{item['id']}_DUP{seen[item['id']]}"
                issue.warnings.append(f"id 重复，已重命名为 {new_id}")
                item["id"] = new_id
            else:
                seen[item["id"]] = 1
            passed.append(item)

        all_issues.append(issue)
        if not issue.ok:
            failed.append(issue)

    logger.info(
        "测试数据校验：共 %d 组，通过 %d 组，丢弃 %d 组",
        len(raws), len(passed), len(failed),
    )
    return passed, all_issues, failed


def describe_issues(issues: list[CaseIssue]) -> str:
    """把校验详情渲染成给人看的文本，用于日志和自修正提示。"""
    lines = [f"- {issue}" for issue in issues if issue.errors or issue.warnings]
    return "\n".join(lines) if lines else "（全部通过，无问题）"
