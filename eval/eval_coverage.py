"""
评测脚本（Day 29 之一）：用例「场景覆盖率」

------------------------------------------------------------------
这是 Day 7 那套评测的"正式交付版"。Day 7 已经能跑评测集，
Day 29 把它单独抽成一个清晰可讲的评测脚本，回答面试问题：
    "你生成的测试用例，覆盖得全不全？怎么量？"
------------------------------------------------------------------
做法：
    评测集 eval/cases/*.yaml 里每条都人工列好了"核心场景清单"
    （比如登录功能必须有：正常登录 / 错误密码 / 空邮箱 ...）。
    让 AI 针对这些功能生成用例，再用关键词匹配，看生成结果
    **命中了哪些预设场景** → 算出"场景覆盖率"。

指标：
    场景覆盖率 = 命中的核心场景数 / 人工预设的核心场景总数

离线说明：
    Mock 模式下假模型不管你问购物车还是搜索都返回同一批登录用例，
    所以除"登录"外覆盖率必然偏低 —— 这是 Mock 的局限，不是 prompt 的问题。
    配好 API Key 后覆盖率才有参考价值（和 Day 7 评测同理）。

用法：
    python eval/eval_coverage.py
    python eval/eval_coverage.py --template <prompt版本>
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from agent import llm_client, structured
from config import settings
from eval import run_eval
from tools.logger import get_logger

logger = get_logger("eval.coverage")


@dataclass
class CoverageRow:
    case_id: str
    feature: str
    structure_rate: float = 0.0       # 结构可用率（生成的用例是不是合法结构）
    coverage_rate: float = 0.0        # 场景覆盖率
    covered: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    cases_valid: int = 0
    cases_total: int = 0
    error: str = ""


@dataclass
class CoverageResult:
    total: int = 0
    avg_structure_rate: float = 0.0
    avg_coverage_rate: float = 0.0
    rows: list[CoverageRow] = field(default_factory=list)
    mode: str = "mock"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    client: llm_client.LLMClient,
    template: str = run_eval.DEFAULT_TEMPLATE,
    eval_cases: list[run_eval.EvalCase] | None = None,
) -> CoverageResult:
    eval_cases = eval_cases if eval_cases is not None else run_eval.load_cases()
    rows: list[CoverageRow] = []

    for ec in eval_cases:
        row = CoverageRow(case_id=ec.id, feature=ec.feature)
        try:
            generated = structured.generate(
                ec.feature, ec.description, template=template, client=client,
            )
            rate, covered, missed = run_eval.compute_coverage(
                generated.cases, ec.core_scenarios
            )
            row.structure_rate = generated.structure_rate
            row.coverage_rate = rate
            row.covered = covered
            row.missed = missed
            row.cases_valid = len(generated.cases)
            row.cases_total = generated.total_raw_cases
        except Exception as exc:  # noqa: BLE001
            logger.exception("用例覆盖率评测 %s 失败", ec.id)
            row.error = str(exc)
        rows.append(row)

    ok = [r for r in rows if not r.error]
    total = len(rows)
    return CoverageResult(
        total=total,
        avg_structure_rate=(sum(r.structure_rate for r in ok) / len(ok)) if ok else 0.0,
        avg_coverage_rate=(sum(r.coverage_rate for r in ok) / len(ok)) if ok else 0.0,
        rows=rows,
        mode="mock" if client.is_mock else "real",
    )


def print_summary(result: CoverageResult) -> None:
    print("\n" + "=" * 70)
    print(f"  用例场景覆盖率评测（模式：{result.mode}）")
    print("=" * 70)
    print(f"  {'功能':<12}{'结构可用率':>12}{'场景覆盖率':>12}{'合格/总数':>12}")
    print("  " + "-" * 64)
    for r in result.rows:
        if r.error:
            print(f"  {r.feature:<12}  评测失败：{r.error[:30]}")
            continue
        print(
            f"  {r.feature:<12}"
            f"{r.structure_rate:>11.0%}"
            f"{r.coverage_rate:>11.0%}"
            f"{f'{r.cases_valid}/{r.cases_total}':>12}"
        )
    print("  " + "-" * 64)
    print(f"  平均结构可用率 : {result.avg_structure_rate:>6.1%}")
    print(f"  平均场景覆盖率 : {result.avg_coverage_rate:>6.1%}")
    print("  目标线         : 结构可用率 ≥ 95%   场景覆盖率 ≥ 80%")
    if result.mode == "mock":
        print("\n  [!] Mock 模式：场景覆盖率不可信（假模型只返回登录用例），"
              "只看结构可用率。")
    print("=" * 70)


def main(template: str = run_eval.DEFAULT_TEMPLATE) -> int:
    settings.ensure_dirs()
    client = llm_client.get_client()

    print("=" * 70)
    print("  用例场景覆盖率评测开始")
    print("=" * 70)
    print(f"  prompt 版本 : {template}")
    print(f"  模式       : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print("=" * 70)

    result = evaluate(client, template=template)
    print_summary(result)

    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = settings.EVAL_DIR / "runs" / f"coverage_{stamp}.json"
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  详细结果：{out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评测 AI 生成用例的场景覆盖率")
    parser.add_argument("--template", default=run_eval.DEFAULT_TEMPLATE, help="prompt 模板名")
    args = parser.parse_args()
    raise SystemExit(main(template=args.template))
