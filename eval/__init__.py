"""
评测模块（Day 7）

目录约定：
    eval/cases/      评测集：固定的需求输入 + 人工预设的核心场景清单
    eval/runs/       每次评测的详细快照（带时间戳，不会被覆盖）
    eval/history.csv 历史汇总，一行一条评测用例，用于画趋势图

小提醒：
    `eval` 恰好是 Python 内置函数的名字，用来当包名看着有点怪，
    但它表意最准，而且在 `from eval import run_eval` 这种用法下
    不会影响内置 eval()，所以保留。
"""
