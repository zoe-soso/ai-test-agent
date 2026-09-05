"""
评测脚本（Day 29 之二）：代码「首次运行通过率」

------------------------------------------------------------------
这脚本量化一个很实在的工程指标：
    "AI 生成的 Playwright 测试代码，第一版就能通过静态检查的比例有多高？
     经过自修正（最多 N 次）后能到多少？"
------------------------------------------------------------------
之所以重要：
    首版通过率 = "模型一次写对的能力"，直接决定你花多少钱、等多久。
    自修正后通过率 = "项目有没有可靠的容错兜底"（写错了能自己改好）。
    两个数字一起看，才说明这套系统"既聪明又稳"。

为什么只用静态检查（validate_code），不真开浏览器？
    1. 静态检查覆盖了最关键的几类错误：语法、POM 规范、是否先开站点、
       调的方法是不是真存在（防模型编方法名）。
    2. 不需要被测网站在线、不需要装浏览器，**任何机器、CI 都能跑**，
       而且确定性好、可复现，适合做"每天跑一次"的回归指标。
    3. 真跑到浏览器是"执行通过率"，那是 Day 27 流水线 + 真实环境的事。

指标：
    首版通过率 = 第一次生成（max_repairs=0）即通过检查的代码数 / 总条数
    修正后通过率 = 允许自修正（max_repairs=N）后通过检查的代码数 / 总条数

用法：
    python eval/eval_code_passrate.py
    python eval/eval_code_passrate.py --max-repairs 2
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

from agent import llm_client, project_profile
from agent.code_generator import CodeGenerator
from agent.models import TestCase
from config import settings
from tools.logger import get_logger

logger = get_logger("eval.code")

# 评测用的"样本用例"：覆盖登录 / 购物车 / 注册三个典型功能。
# 每条都是符合 TestCase 契约的字典（和 Day 3 生成的用例结构一致）。
SAMPLE_CASES: list[tuple[str, TestCase]] = [
    ("用户登录功能", {
        "id": "TC_LOGIN_001", "name": "使用已注册账号正常登录",
        "case_type": "正常", "priority": "P0",
        "steps": ["打开首页", "点击登录入口", "输入正确邮箱", "输入正确密码", "点击登录"],
        "expected": "登录成功，页面显示 Logged in as 用户名",
    }),
    ("购物车结算", {
        "id": "TC_CART_001", "name": "已登录用户将商品加入购物车",
        "case_type": "正常", "priority": "P1",
        "steps": ["打开首页", "搜索商品", "打开商品详情", "点击加入购物车"],
        "expected": "购物车数量 +1",
    }),
    ("注册功能", {
        "id": "TC_REG_001", "name": "用新邮箱完成注册",
        "case_type": "正常", "priority": "P1",
        "steps": ["打开首页", "点击注册", "填写账号信息", "提交"],
        "expected": "注册成功，跳转欢迎页",
    }),
]


@dataclass
class CodePassRow:
    feature: str
    case_id: str
    first_ok: bool = False
    repaired_ok: bool = False
    repairs_used: int = 0
    error: str = ""


@dataclass
class CodePassResult:
    total: int = 0
    first_pass_rate: float = 0.0
    repaired_pass_rate: float = 0.0
    avg_repairs: float = 0.0
    rows: list[CodePassRow] = field(default_factory=list)
    mode: str = "mock"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    client: llm_client.LLMClient,
    profile: project_profile.ProjectProfile | None = None,
    max_repairs: int = 1,
) -> CodePassResult:
    rows: list[CodePassRow] = []
    total_repairs = 0

    for feature, case in SAMPLE_CASES:
        row = CodePassRow(feature=feature, case_id=str(case.get("id", "?")))
        try:
            # 首版：不允许自修正
            first = CodeGenerator(client=client, max_repairs=0, profile=profile).generate(feature, case)
            # 修正版：允许自修正
            repaired = CodeGenerator(client=client, max_repairs=max_repairs, profile=profile).generate(feature, case)
            row.first_ok = first.ok
            row.repaired_ok = repaired.ok
            row.repairs_used = repaired.repairs_used
            total_repairs += repaired.repairs_used
        except Exception as exc:  # noqa: BLE001
            logger.exception("功能 %s 代码评测失败", feature)
            row.error = str(exc)
        rows.append(row)

    total = len(rows)
    first_pass = sum(r.first_ok for r in rows)
    repaired_pass = sum(r.repaired_ok for r in rows)

    return CodePassResult(
        total=total,
        first_pass_rate=(first_pass / total) if total else 0.0,
        repaired_pass_rate=(repaired_pass / total) if total else 0.0,
        avg_repairs=(total_repairs / total) if total else 0.0,
        rows=rows,
        mode="mock" if client.is_mock else "real",
    )


def print_summary(result: CodePassResult) -> None:
    print("\n" + "=" * 70)
    print(f"  代码首版通过率评测（模式：{result.mode}）")
    print("=" * 70)
    print(f"  {'功能':<12}{'首版通过':>10}{'修正后通过':>12}{'自修正次数':>12}")
    print("  " + "-" * 64)
    for r in result.rows:
        print(
            f"  {r.feature:<12}"
            f"{'✓' if r.first_ok else '✗':>10}"
            f"{'✓' if r.repaired_ok else '✗':>12}"
            f"{r.repairs_used:>12}"
        )
    print("  " + "-" * 64)
    print(f"  首版通过率   : {result.first_pass_rate:>6.1%}  "
          f"({sum(r.first_ok for r in result.rows)}/{result.total})")
    print(f"  修正后通过率 : {result.repaired_pass_rate:>6.1%}  "
          f"({sum(r.repaired_ok for r in result.rows)}/{result.total})")
    print(f"  平均自修正   : {result.avg_repairs:.2f} 次/条")
    if result.mode == "mock":
        print("\n  [!] Mock 模式：假模型不真生成 Python 代码，首版通过率偏低是预期。")
        print("      配好 LLM_API_KEY 后才是真实首版通过率。")
    print("=" * 70)


def main(max_repairs: int = 1, profile_name: str | None = None) -> int:
    settings.ensure_dirs()
    client = llm_client.get_client()
    profile = project_profile.load_profile(profile_name)

    print("=" * 70)
    print("  代码首版通过率评测开始")
    print("=" * 70)
    print(f"  样本   : {len(SAMPLE_CASES)} 条（登录 / 购物车 / 注册）")
    print(f"  修正上限: {max_repairs} 次")
    print(f"  模式   : {'Mock（离线）' if client.is_mock else '真实调用'}")
    print("=" * 70)

    result = evaluate(client, profile=profile, max_repairs=max_repairs)
    print_summary(result)

    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    out = settings.EVAL_DIR / "runs" / f"code_passrate_{stamp}.json"
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  详细结果：{out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评测 AI 生成代码的首次运行通过率")
    parser.add_argument("--max-repairs", type=int, default=1, help="最多自修正几次")
    parser.add_argument("--profile", default=None, help="目标项目档案名（Day 28）")
    args = parser.parse_args()
    raise SystemExit(main(max_repairs=args.max_repairs, profile_name=args.profile))
