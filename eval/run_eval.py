"""
评测脚本（Day 7 的核心产出）

------------------------------------------------------------------
为什么 Day 7 就要有这个，而不是等 Day 29
------------------------------------------------------------------
这是我对原计划最大的一条改动建议。原因很简单：

    如果你到 Day 29 才开始想"怎么衡量 AI 生成得好不好"，
    那前面 20 多天你改过的每一个 prompt、生成的每一批用例，
    **都没有留下可对比的基线**。最后只能临时编数字，
    或者临时跑几组数据凑数 —— 面试一追问"这个 85% 怎么测出来的"就虚了。

现在建好，之后每次改 prompt 只需跑一句：

    python main.py eval

结果自动追加到 history.csv。到 Day 29，你要做的只是
**把这二十天的 history.csv 拉个趋势图**。
"我对 AI 输出做过系统性评估"这句话就立住了 ——
绝大多数 AI 项目没有这个，因为他们的作者没想过要测。

------------------------------------------------------------------
三个指标
------------------------------------------------------------------
1. 结构可用率 = 通过校验的用例 / 模型给出的用例总数
     测"模型听不听话"。目标 ≥ 95%。
     分母为 0（连 JSON 都没 parse 出来）时记 0 分，不给自己放水。

2. 场景覆盖率 = 生成用例命中的核心场景 / 人工预设的场景清单
     测"生成得全不全"。目标 ≥ 80%。
     场景清单是人工写在 eval/cases/*.yaml 里的，关键词匹配，完全可复现。

3. 单次成本 = 一条需求生成全套用例花了多少 token / 多少钱
     测"划不划算"。这是工程约束，不是技术炫技。

------------------------------------------------------------------
怎么用
------------------------------------------------------------------
    python eval/run_eval.py                    # 用默认 prompt 跑全部评测集
    python eval/run_eval.py --template xxx     # 换 prompt 版本跑
    python main.py eval                        # 等价的命令行入口

每次跑完会做两件事：
    - 详细快照存进 eval/runs/<时间戳>.json
    - 一行汇总追加进 eval/history.csv
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# eval/ 目录不在项目根的导入路径里（直接 `python eval/run_eval.py` 时），
# 所以先把项目根塞进 sys.path，否则 import agent / config / prompts 会失败。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import llm_client, structured  # noqa: E402
from agent.models import DESIGN_METHODS, TestCase  # noqa: E402
from config import settings  # noqa: E402
from tools import file_io  # noqa: E402
from tools.logger import get_logger  # noqa: E402

logger = get_logger("eval")

DEFAULT_TEMPLATE = structured.DEFAULT_TEMPLATE
HISTORY_FIELDS = [
    "date", "prompt_version", "case_id",
    "structure_rate", "coverage_rate", "method_coverage",
    "cases_valid", "cases_total",
    "total_tokens", "cost_yuan",
    "repairs", "attempts", "mode",
]


# ----------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------
@dataclass
class EvalCase:
    """一条评测输入（从 eval/cases/*.yaml 读进来）。"""

    id: str
    feature: str
    description: str
    core_scenarios: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CaseResult:
    """一条评测输入的跑分结果。"""

    case_id: str
    structure_rate: float = 0.0
    coverage_rate: float = 0.0
    method_coverage: float = 0.0          # Day 10：设计方法覆盖率
    methods_used: list[str] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    cases_valid: int = 0
    cases_total: int = 0
    total_tokens: int = 0
    cost_yuan: float = 0.0
    repairs: int = 0
    attempts: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ----------------------------------------------------------------------
# 评测集加载
# ----------------------------------------------------------------------
def load_cases(directory: Path | None = None) -> list[EvalCase]:
    """加载 eval/cases/ 下所有 yaml 评测输入。"""
    directory = directory or (settings.EVAL_DIR / "cases")
    if not directory.exists():
        raise FileNotFoundError(f"评测集目录不存在：{directory}")

    cases: list[EvalCase] = []
    for path in sorted(directory.glob("*.yaml")):
        raw = file_io.read_yaml(path)
        if not isinstance(raw, dict):
            logger.warning("评测文件格式不对，跳过：%s", path.name)
            continue
        cases.append(
            EvalCase(
                id=str(raw.get("id") or path.stem),
                feature=str(raw.get("feature") or path.stem),
                description=str(raw.get("description") or "").strip(),
                core_scenarios=list(raw.get("core_scenarios") or []),
            )
        )

    logger.info("加载评测集：%d 条（来自 %s）", len(cases), directory)
    return cases


# ----------------------------------------------------------------------
# 指标计算
# ----------------------------------------------------------------------
def compute_coverage(
    cases: list[TestCase],
    scenarios: list[dict[str, Any]],
) -> tuple[float, list[str], list[str]]:
    """算场景覆盖率。

    判定：某条用例的 name / expected / steps 里，
    出现了该场景 keywords 中的任意一个词，就算覆盖了。
    比较前统一转小写，避免 Logged in as / logged in as 这种大小写差异漏判。

    返回 (覆盖率, 已覆盖场景名, 未覆盖场景名)。
    """
    if not scenarios:
        return 0.0, [], []

    texts = [
        " ".join([case["name"], case["expected"], *case["steps"]]).lower()
        for case in cases
    ]

    covered: list[str] = []
    missed: list[str] = []

    for scenario in scenarios:
        name = str(scenario.get("name", "未命名场景"))
        keywords = [str(kw).lower() for kw in scenario.get("keywords", [])]

        hit = any(keyword in text for text in texts for keyword in keywords)
        (covered if hit else missed).append(name)

    rate = len(covered) / len(scenarios)
    return rate, covered, missed


def compute_method_coverage(cases: list[TestCase]) -> tuple[float, list[str]]:
    """算设计方法覆盖率（Day 10 新增的指标）。

        用到了几种设计方法 / 一共 5 种

    "未标注"会被剔除 —— 模型没写就不该给分，否则指标会虚高。

    为什么这个指标值钱：
        它测的是纯粹的**测试专业度**，和 AI 技术无关。
        "我的 AI 生成的用例覆盖了 5 种设计方法中的 4 种"——
        这句话别的 AI 项目说不出来，因为他们的作者不懂测试。
        而这正是你相对其他 AI 项目从业者的优势。
    """
    used = {
        case.get("design_method")
        for case in cases
        if case.get("design_method")
    }
    used.discard("未标注")

    rate = len(used) / len(DESIGN_METHODS) if DESIGN_METHODS else 0.0
    return rate, sorted(used)


def _usage_delta_cost(before: tuple[int, int], after: tuple[int, int]) -> tuple[int, float]:
    """根据 usage 快照的差值，算这次生成消耗了多少 token 和多少钱。"""
    delta_prompt = max(0, after[0] - before[0])
    delta_completion = max(0, after[1] - before[1])
    cost = (
        delta_prompt / 1_000_000 * settings.PRICE_INPUT_PER_1M
        + delta_completion / 1_000_000 * settings.PRICE_OUTPUT_PER_1M
    )
    return delta_prompt + delta_completion, cost


# ----------------------------------------------------------------------
# 跑评测
# ----------------------------------------------------------------------
def run_all(
    template: str = DEFAULT_TEMPLATE,
    eval_cases: list[EvalCase] | None = None,
    client: llm_client.LLMClient | None = None,
    max_repairs: int = 1,
) -> list[CaseResult]:
    """跑一遍完整评测集。

    一条挂了不影响其他条 —— 每条单独 try，
    因为评测最怕的是"有一个 case 炸了，整轮白跑"。
    """
    client = client or llm_client.get_client()
    eval_cases = eval_cases if eval_cases is not None else load_cases()

    results: list[CaseResult] = []

    for index, eval_case in enumerate(eval_cases, start=1):
        print(f"  [{index}/{len(eval_cases)}] {eval_case.feature} ... ", end="", flush=True)

        before = (client.usage.prompt_tokens, client.usage.completion_tokens)

        try:
            generated = structured.generate(
                eval_case.feature,
                eval_case.description,
                template=template,
                max_repairs=max_repairs,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001 - 评测要跑完全集，不能因一条中断
            logger.exception("评测用例 %s 生成失败", eval_case.id)
            print(f"失败（{type(exc).__name__}）")
            results.append(CaseResult(case_id=eval_case.id, error=str(exc)))
            continue

        after = (client.usage.prompt_tokens, client.usage.completion_tokens)
        tokens, cost = _usage_delta_cost(before, after)

        rate, covered, missed = compute_coverage(
            generated.cases, eval_case.core_scenarios
        )
        method_rate, methods_used = compute_method_coverage(generated.cases)

        results.append(
            CaseResult(
                case_id=eval_case.id,
                structure_rate=generated.structure_rate,
                coverage_rate=rate,
                method_coverage=method_rate,
                methods_used=methods_used,
                covered=covered,
                missed=missed,
                cases_valid=len(generated.cases),
                cases_total=generated.total_raw_cases,
                total_tokens=tokens,
                cost_yuan=cost,
                repairs=generated.repairs_used,
                attempts=generated.attempts,
            )
        )

        print(
            f"结构 {generated.structure_rate:>5.0%} | "
            f"覆盖 {rate:>5.0%} | 方法 {method_rate:>5.0%} | "
            f"{tokens:>5} tokens | {settings.CURRENCY}{cost:.4f}"
        )

    return results


# ----------------------------------------------------------------------
# 结果落盘
# ----------------------------------------------------------------------
def save_run(results: list[CaseResult], template: str, mode: str) -> Path:
    """存一份详细快照到 eval/runs/。

    带时间戳而不是只用日期：同一天改了三次 prompt 就要跑三次，
    只用日期会互相覆盖，那趋势图就没法画了。
    """
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    path = settings.EVAL_DIR / "runs" / f"{stamp}.json"

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_version": template,
        "mode": mode,
        "results": [asdict(r) for r in results],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("评测快照已保存：%s", path)
    return path


def append_history(results: list[CaseResult], template: str, mode: str) -> Path:
    """把这一轮结果追加进 history.csv（不存在就先写表头）。"""
    path = settings.EVAL_DIR / "history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    need_header = (not path.exists()) or path.stat().st_size == 0
    today = time.strftime("%Y-%m-%d")

    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if need_header:
            writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "date": today,
                    "prompt_version": template,
                    "case_id": result.case_id,
                    "structure_rate": f"{result.structure_rate:.4f}",
                    "coverage_rate": f"{result.coverage_rate:.4f}",
                    "method_coverage": f"{result.method_coverage:.4f}",
                    "cases_valid": result.cases_valid,
                    "cases_total": result.cases_total,
                    "total_tokens": result.total_tokens,
                    "cost_yuan": f"{result.cost_yuan:.6f}",
                    "repairs": result.repairs,
                    "attempts": result.attempts,
                    "mode": mode,
                }
            )

    logger.info("已追加 %d 行到 %s", len(results), path)
    return path


# ----------------------------------------------------------------------
# 展示
# ----------------------------------------------------------------------
def print_summary(results: list[CaseResult], template: str, mode: str = "mock") -> None:
    """打印汇总表 + 总评。"""
    ok_results = [r for r in results if r.ok]
    if not ok_results:
        print("\n全部评测用例都失败了，没有可统计的结果。")
        return

    print("\n" + "=" * 74)
    print(f"  评测汇总（prompt 版本：{template}）")
    print("=" * 74)
    print(f"  {'用例':<12}{'结构可用率':>12}{'场景覆盖率':>12}"
          f"{'合格/总数':>12}{'tokens':>10}{'成本':>12}")
    print("  " + "-" * 70)

    for result in ok_results:
        print(
            f"  {result.case_id:<12}"
            f"{result.structure_rate:>11.0%} "
            f"{result.coverage_rate:>11.0%} "
            f"{f'{result.cases_valid}/{result.cases_total}':>12}"
            f"{result.total_tokens:>10}"
            f"{settings.CURRENCY}{result.cost_yuan:>11.4f}"
        )

    print("  " + "-" * 70)

    avg_structure = sum(r.structure_rate for r in ok_results) / len(ok_results)
    avg_coverage = sum(r.coverage_rate for r in ok_results) / len(ok_results)
    total_tokens = sum(r.total_tokens for r in ok_results)
    total_cost = sum(r.cost_yuan for r in ok_results)
    total_repairs = sum(r.repairs for r in ok_results)

    print(f"  {'平均':<12}{avg_structure:>11.0%} {avg_coverage:>11.0%}"
          f"{'':>12}{total_tokens:>10}{settings.CURRENCY}{total_cost:>11.4f}")

    print("\n  目标线：结构可用率 ≥ 95%   场景覆盖率 ≥ 80%")
    print(f"  自修正触发：{total_repairs} 次（说明模型第一次输出就有 {total_repairs} 次不合规）")

    if mode == "mock":
        # 这条提示非常重要：不加的话，你会以为 prompt 写得很烂
        print(
            "\n  [!] Mock 模式下「场景覆盖率」不可信，只看「结构可用率」。\n"
            "      原因：假模型不管你问购物车还是搜索，都返回同一批登录用例，\n"
            "      所以除 login 外覆盖率必然偏低。这不是 prompt 的问题。\n"
            "      配上真实 API Key 后，覆盖率才有参考价值。"
        )

    # 未覆盖的场景列出来，这是改进 prompt 最直接的输入
    all_missed: list[str] = []
    for result in ok_results:
        for name in result.missed:
            all_missed.append(f"{result.case_id}/{name}")
    if all_missed:
        print("\n  未覆盖的场景（改 prompt 时优先补这些）：")
        for item in all_missed[:12]:
            print(f"    - {item}")
        if len(all_missed) > 12:
            print(f"    ... 还有 {len(all_missed) - 12} 个")

    print("=" * 74)


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main(template: str = DEFAULT_TEMPLATE, max_repairs: int = 1) -> int:
    settings.ensure_dirs()
    client = llm_client.get_client()
    client.usage.reset()

    mode = "mock" if client.is_mock else "real"

    print("=" * 74)
    print("  评测开始")
    print("=" * 74)
    print(f"  prompt 版本 : {template}")
    print(f"  运行模式    : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print(f"  模型        : {client.model}")
    print("=" * 74)

    results = run_all(template=template, client=client, max_repairs=max_repairs)

    print_summary(results, template, mode)

    run_path = save_run(results, template, mode)
    history_path = append_history(results, template, mode)

    print(f"\n  快照  : {run_path}")
    print(f"  历史  : {history_path}")
    print(f"  总消耗: {client.usage}\n")

    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="跑一遍 AI 测试用例生成评测集")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="prompt 模板名")
    parser.add_argument("--max-repairs", type=int, default=1, help="最多自修正几次")
    args = parser.parse_args()

    sys.exit(main(template=args.template, max_repairs=args.max_repairs))
