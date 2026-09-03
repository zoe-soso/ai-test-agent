"""
测试数据生成模块（Day 12）

计划要求：
    登录：username / password
    让 AI 生成：正确数据 / 错误密码 / 空密码 / 超长密码 / 特殊字符 / 不存在账号
    输出 YAML

------------------------------------------------------------------
这一天真正要理解的事：为什么数据和用例必须分开
------------------------------------------------------------------
先想一个问题：如果用例和数据写在一起，会发生什么？

    用例1：输入 user@test.com / Test@123，点击登录，预期成功
    用例2：输入 user@test.com / wrongpass，点击登录，预期失败
    用例3：输入 user@test.com / （空），点击登录，预期失败

这三条的步骤**一模一样**，只有数据不同。
换个环境（测试服的数据库里没有 user@test.com 这个账号）就全废了，
你得改三个地方。

分开之后变成：

    用例（只写一遍）：输入 {username} / {password}，点击登录，预期 {expected}
    数据（随便加）：  10 组、100 组，改数据不用碰用例

这就是**数据驱动测试（DDT）**。
你的 ecommerce-test-automation 项目里 data/ 目录干的就是这件事，
现在只不过把"手写数据"换成"让 AI 生成数据"。

------------------------------------------------------------------
AI 生成数据的三个坑（本模块逐个处理）
------------------------------------------------------------------
1. **生成"占位符"而不是真值**
   模型最爱给 `{"password": "正确的密码"}` 这种。
   看着像数据，其实没法执行。
   → 处理：prompt 里明确禁止 + 校验空值/描述性内容

2. **超长数据不够长**
   你让它生成"超长密码"，它给你 30 个字符就交差了 —— 那测不出边界。
   → 处理：prompt 要求给出真实长度，并在报告里统计实际长度

3. **六类数据覆盖不全**
   模型容易生成一堆"正确数据"凑数，异常类随便给两条。
   → 处理：和 Day 9 用例生成一样，做**覆盖自检 + 定向补充**
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent import json_utils, llm_client, validator
from agent.models import DATA_TYPES, TestCase, TestData, TestDataSuite
from agent.validator import CaseIssue
from config import settings
from prompts import loader
from tools.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TEMPLATE = "testdata"
# 注意：不能用用例的 fix_json。那个模板强制要求 case_type/priority/steps 字段，
# 拿去修测试数据会直接把模型带偏。格式修复模板必须和数据契约一一对应。
FIX_TEMPLATE = "fix_json_data"

# 数据类型 → 用例类型 的对应关系。
# 用来把数据挂到用例上，让最终 YAML 里能看出"这组数据给哪条用例用"。
DATA_TYPE_TO_CASE_TYPE: dict[str, str] = {
    "正确数据": "正常",
    "错误数据": "异常",
    "不存在数据": "异常",
    "空值": "边界",
    "超长数据": "边界",
    "特殊字符": "边界",
}

# 用来识别"这值是不是占位符而不是真数据"。
# 模型偷懒时最爱写这些，看着像数据，跑起来全是 bug。
PLACEHOLDER_WORDS = (
    "正确", "错误", "有效", "无效", "xxx", "test", "示例",
    "例如", "待填", "todo", "待定", "某个", "某值", "任意",
)

# 子串匹配时允许的最大长度。
#
# 这个值很关键，调大调小都会出问题（Day 12 两个方向都踩过）：
#     太大（比如 20）→ "testuser" 这种正常用户名会被误判成占位符
#     太小（比如 3）→ "正确的密码" 这种真正的占位符就漏掉了
#
# 取 6 的依据：真正的占位符通常很短（"正确""test""xxx"），
# 而合法数据哪怕包含这些词（testuser / Test@123456）也普遍超过 6 个字符。
PLACEHOLDER_SUBSTRING_MAX_LEN = 6

# 判定为"超长数据"的最小长度。低于这个数说明模型没真的给长数据
MIN_OVERLONG_LENGTH = 100

# ----------------------------------------------------------------------
# 超长数据的"紧凑标记法"
# ----------------------------------------------------------------------
# 为什么不让模型直接把 500 个字符写出来？Day 12 实测踩出来的三个坑：
#
#   1. **模型不会数数**
#      你让它写 500 个字符，它可能给你 480 个或 530 个。
#      而"超长"边界测试的价值恰恰在于**长度是确定的** ——
#      差几十个字符，测的就不是你想测的那个边界了。
#
#   2. **贵，而且会写爆输出预算**
#      500 个字符大约几百个 token。几组超长数据就把整个输出额度吃光。
#      Day 12 实测：deepseek-chat 在 8192 的上限下仍然被写满截断，
#      JSON 只输出了一半，表现为"解析失败"，极难定位。
#
#   3. **可能陷入重复生成**
#      逐字符输出时模型偶尔会开始循环输出同一个字符。
#
# 正确分工：
#     模型负责语义 —— "这里需要一个 500 长的、由 A 组成的密码"
#     Python 负责机械 —— 精确地生成 500 个 A
#
# 这和 Day 11 "能用规则解决的不要问 LLM" 是同一条原则：
# **别让 LLM 做它不擅长的事。**
LONG_MARKER_RE = re.compile(r"<<LONG:(\d+):(.*?)>>", re.DOTALL)

# 长度上限兜底：防止模型写 <<LONG:99999999:A>> 把内存撑爆
MAX_LONG_LENGTH = 10_000


def expand_markers(value: Any) -> Any:
    """把字符串里的 <<LONG:500:A>> 展开成真正的 500 个 A。

    非字符串原样返回（数字、布尔、null 等不需要展开）。
    """
    if not isinstance(value, str):
        return value

    def _expand(match: re.Match) -> str:
        length = min(int(match.group(1)), MAX_LONG_LENGTH)
        char = match.group(2) or "A"
        return char * length

    return LONG_MARKER_RE.sub(_expand, value)


def expand_data_markers(data: list[TestData]) -> list[TestData]:
    """对一批数据的所有字段值做标记展开。"""
    for item in data:
        item["fields"] = {
            key: expand_markers(value) for key, value in item["fields"].items()
        }
    return data


@dataclass
class DataGenReport:
    """一次数据生成的过程数据（评测和命令行都要看）。"""

    feature: str
    attempts: int = 0            # LLM 调用次数
    repairs_used: int = 0        # 自修正次数
    total_raw: int = 0           # 模型一共给了多少组
    valid: int = 0               # 合格多少组
    missing_types: list[str] = field(default_factory=list)
    supplemented: int = 0        # 定向补充拿到多少组
    linked: int = 0              # 成功关联到用例的组数

    @property
    def structure_rate(self) -> float:
        if self.total_raw <= 0:
            return 0.0
        return self.valid / self.total_raw

    @property
    def type_coverage(self) -> float:
        """六类数据的覆盖率。"""
        if not DATA_TYPES:
            return 0.0
        return (len(DATA_TYPES) - len(self.missing_types)) / len(DATA_TYPES)

    def to_dict(self) -> dict[str, Any]:
        return {
            "功能": self.feature,
            "调用次数": self.attempts,
            "自修正次数": self.repairs_used,
            "模型给出组数": self.total_raw,
            "合格组数": self.valid,
            "结构可用率": f"{self.structure_rate:.0%}",
            "数据类型覆盖": f"{self.type_coverage:.0%}",
            "仍缺类型": "、".join(self.missing_types) if self.missing_types else "无",
            "定向补充": self.supplemented,
            "已关联用例": self.linked,
        }


class TestDataGenerator:
    """测试数据生成器。"""

    def __init__(
        self,
        client: llm_client.LLMClient | None = None,
        template: str = DEFAULT_TEMPLATE,
        max_repairs: int = 1,
    ) -> None:
        self.client = client or llm_client.get_client()
        self.template = template
        self.max_repairs = max_repairs

    # ------------------------------------------------------------------
    def generate(
        self,
        feature: str,
        description: str = "（无额外描述）",
        params: list[str] | None = None,
        cases: list[TestCase] | None = None,
    ) -> tuple[TestDataSuite, DataGenReport]:
        """生成测试数据。

        参数：
            feature      被测功能名
            description  需求描述
            params       参数名清单，如 ["username","password"]；
                         不传就让模型自己判断需要哪些参数
            cases        可选。传了就把数据关联到对应用例上
        """
        report = DataGenReport(feature=feature)
        effective_params = params

        # ---- 1. 生成 ----
        raw = self._call(feature, description, effective_params)
        report.attempts = 1

        data, issues, failed, total = self._parse(raw, effective_params)
        report.total_raw, report.valid = total, len(data)

        # ---- 2. 自修正（和 Day 7 用例生成完全同构）----
        while failed and report.repairs_used < self.max_repairs:
            report.repairs_used += 1
            report.attempts += 1

            errors = validator.describe_issues(failed)
            fix = loader.load(FIX_TEMPLATE, errors=errors, raw=raw)
            raw = self.client.chat_messages(
                fix.to_messages(),
                max_tokens=settings.LLM_MAX_TOKENS_DATA,
                mock_hint="testdata",
            )

            data, issues, failed, total = self._parse(raw, effective_params)
            report.total_raw = max(report.total_raw, total)
            report.valid = len(data)

        # ---- 3. 覆盖自检 + 定向补充 ----
        report.missing_types = self.check_coverage(data)
        if report.missing_types:
            logger.info("缺少数据类型：%s，尝试定向补充", "、".join(report.missing_types))
            added = self._supplement(feature, description, effective_params,
                                     data, report.missing_types)
            if added:
                report.supplemented = len(added)
                data = data + added
                report.valid = len(data)
                report.missing_types = self.check_coverage(data)

        # ---- 4. 参数名：模型没给就用它实际返回的第一组数据的 key ----
        if not effective_params and data:
            effective_params = list(data[0]["fields"].keys())

        # ---- 5. 关联用例 ----
        if cases:
            report.linked = self.link_to_cases(data, cases)

        suite: TestDataSuite = {
            "feature": feature,
            "params": effective_params or [],
            "data": data,
        }
        return suite, report

    # ------------------------------------------------------------------
    # 内部：调用 / 解析 / 补充 / 关联
    # ------------------------------------------------------------------
    def _call(self, feature: str, description: str,
              params: list[str] | None) -> str:
        """调一次 LLM。"""
        param_text = "、".join(params) if params else "（请根据功能自行判断需要哪些参数）"
        prompt = loader.load(
            self.template,
            feature=feature,
            description=description,
            params=param_text,
        )
        # 数据生成单独放宽输出上限（原因见 settings.LLM_MAX_TOKENS_DATA 的注释）
        return self.client.chat_messages(
            prompt.to_messages(),
            max_tokens=settings.LLM_MAX_TOKENS_DATA,
            mock_hint="testdata",
        )

    def _parse(
        self, raw: str, params: list[str] | None
    ) -> tuple[list[TestData], list[CaseIssue], list[CaseIssue], int]:
        """解析 + 校验一步到位。"""
        try:
            payload = json_utils.extract_json(raw)
        except Exception as exc:  # noqa: BLE001
            issue = CaseIssue(index=0, case_id="<解析失败>", errors=[str(exc)])
            return [], [issue], [issue], 0

        raws = _extract_data_items(payload)
        passed, all_issues, failed = validator.validate_data(raws, params=params)
        # 展开 <<LONG:n:c>> 标记，得到真正长度的数据
        passed = expand_data_markers(passed)
        return passed, all_issues, failed, len(raws)

    def _supplement(
        self,
        feature: str,
        description: str,
        params: list[str] | None,
        existing: list[TestData],
        missing: list[str],
    ) -> list[TestData]:
        """只补缺的那几类数据，不全量重生成。

        和 Day 9 用例补充是同一个思路：
            全量重生成 = 丢弃已有成果 + 花双倍的钱
            定向补充   = 只花小钱，已有的都保住
        """
        param_text = "、".join(params) if params else "（同上）"
        prompt = loader.load(
            "supplement_data",
            feature=feature,
            description=description,
            params=param_text,
            missing="、".join(missing),
            existing=_render_existing(existing),
        )

        try:
            raw = self.client.chat_messages(
                prompt.to_messages(),
                max_tokens=settings.LLM_MAX_TOKENS_DATA,
                mock_hint="testdata",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("补充测试数据失败，保留已有数据：%s", exc)
            return []

        try:
            payload = json_utils.extract_json(raw)
            raws = _extract_data_items(payload)
            passed, _, _ = validator.validate_data(raws, params=params)
            passed = expand_data_markers(passed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("补充数据解析失败，保留已有数据：%s", exc)
            return []

        # 只保留真正缺的那些类型，防止模型又生成一堆重复的
        needed = set(missing)
        picked = [d for d in passed if d["data_type"] in needed]

        # 同一类型内去重（id 不重复即可）
        seen = {d["id"] for d in existing}
        unique = []
        for item in picked:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            unique.append(item)

        if unique:
            logger.info("定向补充到 %d 组数据：%s",
                        len(unique), "、".join(d["data_type"] for d in unique))
        return unique

    # ------------------------------------------------------------------
    # 对外：覆盖自检 / 质量检查 / 关联 / 保存
    # ------------------------------------------------------------------
    def check_coverage(self, data: list[TestData]) -> list[str]:
        """检查六类数据是否齐全，返回缺失的类型。"""
        present = {d["data_type"] for d in data}
        return [t for t in DATA_TYPES if t not in present]

    def quality_check(self, data: list[TestData]) -> list[str]:
        """数据质量检查：揪出占位符和"不够长"的超长数据。

        这是纯规则检查，不花钱。返回问题列表。
        """
        problems: list[str] = []

        for item in data:
            for key, value in item["fields"].items():
                text = str(value)
                lowered = text.lower()

                # 占位符：值本身就是描述性词语。
                # 精确匹配无条件判定；子串匹配只在值很短时才算，
                # 否则 "testuser" 这种合法用户名会被误伤。
                if lowered in PLACEHOLDER_WORDS or (
                    any(w in lowered for w in PLACEHOLDER_WORDS)
                    and len(text) <= PLACEHOLDER_SUBSTRING_MAX_LEN
                ):
                    problems.append(
                        f"{item['id']} 的 {key} 像占位符而非真实数据：「{text}」"
                    )

            # 超长数据：只要求"至少有一个字段"真的够长。
            #
            # 为什么不是每个字段都要长？
            #     超长测试的正确做法是**单一变量** ——
            #     只让被测的那个字段超长，其余字段保持正常值。
            #     如果一组数据里 username 和 password 都改成 500 字符，
            #     测失败了你根本不知道是哪个字段的锅。
            #     所以这里检测"有没有一个字段达标"，而不是"每个字段都达标"。
            if item["data_type"] == "超长数据" and item["fields"]:
                lengths = [len(str(v)) for v in item["fields"].values()]
                if not any(n >= MIN_OVERLONG_LENGTH for n in lengths):
                    problems.append(
                        f"{item['id']} 标为超长数据，但没有任何字段达到 "
                        f"{MIN_OVERLONG_LENGTH} 字符（最长仅 {max(lengths)} 字符）"
                    )

        return problems

    def link_to_cases(self, data: list[TestData], cases: list[TestCase]) -> int:
        """把数据挂到用例上（按类型匹配，同类型内轮流分配）。

        关联的价值：拿到 YAML 之后，能直接看出
        "这组数据应该喂给哪条用例"，执行时不用再人肉配对。
        """
        pools: dict[str, list[TestCase]] = {}
        for case in cases:
            pools.setdefault(case["case_type"], []).append(case)

        cursors: dict[str, int] = {}
        linked = 0

        for item in data:
            want = DATA_TYPE_TO_CASE_TYPE.get(item["data_type"])
            pool = pools.get(want or "", [])
            if not pool:
                continue
            cursor = cursors.get(want or "", 0)
            # 同类型用例轮流分，避免所有数据都压在第一条用例上
            item["link_case"] = pool[cursor % len(pool)]["id"]
            cursors[want or ""] = cursor + 1
            linked += 1

        return linked

    def save(self, suite: TestDataSuite, out_path: str | None = None) -> str:
        """落盘 YAML。"""
        from tools import file_io

        target = Path(out_path) if out_path else settings.OUTPUT_DIR / "testdata.yaml"
        file_io.write_yaml(target, suite)
        logger.info("测试数据已写入：%s", target)
        return str(target)


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _extract_data_items(payload: Any) -> list[dict[str, Any]]:
    """从解析后的 JSON 里取出数据列表。

    模型对外层结构依然很随意，可能是：
        {"data": [...]}
        {"test_data": [...]}
        {"params": [...], "data": [...]}
        [ {...}, {...} ]
    """
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("data", "test_data", "testdata", "test_data_list", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

        lists = [v for v in payload.values() if isinstance(v, list)]
        # params 是字符串列表，不是数据列表，要排掉
        lists = [
            v for v in lists
            if all(isinstance(x, dict) for x in v) and v
        ]
        if len(lists) == 1:
            return lists[0]

    raise ValueError(f"JSON 里找不到测试数据列表，实际类型：{type(payload).__name__}")


def _render_existing(existing: list[TestData]) -> str:
    """把已有数据渲染成"已有清单"，让模型知道别再生成重复的。"""
    if not existing:
        return "（暂无）"
    return "\n".join(
        f"- {d['id']} {d['name']}（{d['data_type']}）" for d in existing
    )
