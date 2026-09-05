"""
评测脚本（Day 29 之三）：AI 缺陷分析准确率

------------------------------------------------------------------
这脚本解决一个面试必问的问题：
    "你说你的 Agent 能分析失败原因，那你**怎么证明它分析得对**？"
------------------------------------------------------------------
办法：准备一份人工标注的失败样本（eval/defect_labels.yaml），
每条都写好了"正确分类 / 正确严重程度"。把这些样本丢给 AI 缺陷分析器，
看它判出来的 category / severity 和人工标注对不对得上，算出准确率。

指标：
    1. 分类准确率   = 分类判对的样本数 / 总样本数
    2. 严重度准确率 = 严重程度判对的样本数 / 总样本数
    3. 分类有效率   = 落在白名单内的样本数 / 总样本数（不归"未知"的比例）

离线说明：
    Mock 模式下分析器永远返回固定的「元素定位问题/P1」，
    所以只有标注为这个的样本会对 —— 这**如实**反映了"没接真模型时本指标没意义"。
    配好 API Key 后，这里就是真刀真枪的准确率。

用法：
    python eval/eval_defect_accuracy.py
    python eval/eval_defect_accuracy.py --labels eval/defect_labels.yaml
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

from agent import llm_client
from agent.defect_analyzer import DefectAnalyzer
from agent.failure_collector import FailureRecord
from config import settings
from tools import file_io
from tools.logger import get_logger

logger = get_logger("eval.defect")

DEFAULT_LABELS = settings.EVAL_DIR / "defect_labels.yaml"


@dataclass
class DefectEvalRow:
    sample_id: str
    test_name: str
    predicted_category: str = ""
    expected_category: str = ""
    category_ok: bool = False
    predicted_severity: str = ""
    expected_severity: str = ""
    severity_ok: bool = False
    classified: bool = False          # 预测分类是否落在白名单
    error: str = ""


@dataclass
class DefectEvalResult:
    total: int = 0
    category_accuracy: float = 0.0
    severity_accuracy: float = 0.0
    classify_rate: float = 0.0
    rows: list[DefectEvalRow] = field(default_factory=list)
    mode: str = "mock"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_labels(path: Path) -> list[dict[str, Any]]:
    raw = file_io.read_yaml(path)
    if not isinstance(raw, list):
        raise ValueError(f"标注文件应为 YAML 列表：{path}")
    return raw


def evaluate(
    client: llm_client.LLMClient,
    labels: list[dict[str, Any]],
) -> DefectEvalResult:
    """跑完整个标注集，返回准确率结果。"""
    analyzer = DefectAnalyzer(client)
    rows: list[DefectEvalRow] = []

    for item in labels:
        row = DefectEvalRow(
            sample_id=str(item.get("id", "?")),
            test_name=str(item.get("test_name", "")),
            expected_category=str(item.get("expected_category", "")),
            expected_severity=str(item.get("expected_severity", "")),
        )
        try:
            failure = FailureRecord(
                test_name=row.test_name,
                error=str(item.get("error", "")),
                traceback=str(item.get("traceback", item.get("error", ""))),
            )
            analysis = analyzer.analyze(failure)
            row.predicted_category = analysis.category
            row.predicted_severity = analysis.severity
            row.classified = analysis.is_classified
            row.category_ok = analysis.category == row.expected_category
            row.severity_ok = analysis.severity == row.expected_severity
        except Exception as exc:  # noqa: BLE001
            logger.exception("样本 %s 分析失败", row.sample_id)
            row.error = str(exc)
        rows.append(row)

    ok_cat = sum(r.category_ok for r in rows)
    ok_sev = sum(r.severity_ok for r in rows)
    classified = sum(r.classified for r in rows)
    total = len(rows)

    return DefectEvalResult(
        total=total,
        category_accuracy=(ok_cat / total) if total else 0.0,
        severity_accuracy=(ok_sev / total) if total else 0.0,
        classify_rate=(classified / total) if total else 0.0,
        rows=rows,
        mode="mock" if client.is_mock else "real",
    )


def print_summary(result: DefectEvalResult) -> None:
    print("\n" + "=" * 70)
    print(f"  缺陷分析准确率评测（模式：{result.mode}）")
    print("=" * 70)
    print(f"  {'样本':<6}{'预测分类':>12}{'预期分类':>12}{'分类对':>6}{'严重度对':>8}")
    print("  " + "-" * 64)
    for r in result.rows:
        print(
            f"  {r.sample_id:<6}"
            f"{r.predicted_category:>12}{r.expected_category:>12}"
            f"{'✓' if r.category_ok else '✗':>6}"
            f"{'✓' if r.severity_ok else '✗':>8}"
        )
    print("  " + "-" * 64)
    print(f"  分类准确率   : {result.category_accuracy:>6.1%}  "
          f"({sum(r.category_ok for r in result.rows)}/{result.total})")
    print(f"  严重度准确率 : {result.severity_accuracy:>6.1%}  "
          f"({sum(r.severity_ok for r in result.rows)}/{result.total})")
    print(f"  分类有效率   : {result.classify_rate:>6.1%}  "
          f"(不归'未知'的比例)")
    if result.mode == "mock":
        print("\n  [!] Mock 模式：分析器固定返回『元素定位问题/P1』，"
              "准确率仅供参考。")
        print("      配好 LLM_API_KEY 后才是真实准确率。")
    print("=" * 70)


def main(labels_path: str | None = None) -> int:
    settings.ensure_dirs()
    path = Path(labels_path) if labels_path else DEFAULT_LABELS
    client = llm_client.get_client()

    print("=" * 70)
    print("  缺陷分析准确率评测开始")
    print("=" * 70)
    print(f"  标注集 : {path}")
    print(f"  模式   : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print("=" * 70)

    labels = load_labels(path)
    result = evaluate(client, labels)
    print_summary(result)

    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = settings.EVAL_DIR / "runs" / f"defect_accuracy_{stamp}.json"
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  详细结果：{out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评测 AI 缺陷分析的准确率")
    parser.add_argument("--labels", default=None, help="标注集 YAML 路径")
    args = parser.parse_args()
    raise SystemExit(main(labels_path=args.labels))
