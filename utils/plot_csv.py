"""把训练日志 CSV（长格式：t_env,name,value）转成折线图。

每个不同的 ``name`` 是一条曲线，x 轴是 ``t_env``，y 轴是 ``value``。

参数从 yaml 读取（默认 ``config/plot.yaml``），命令行参数可选覆盖。

用法示例::

    # 用 config/plot.yaml 里的配置
    python utils/plot_csv.py

    # 指定另一份配置
    python utils/plot_csv.py --config config/plot.yaml

    # 用命令行覆盖部分字段（其余仍取 yaml）
    python utils/plot_csv.py --csv logs/3m_s1/stats.csv --metrics return_mean,battle_won_mean,loss -o curves.png
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境（如服务器/WSL）也能直接出图
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def load_series(path: Path) -> dict[str, list[tuple[float, float]]]:
    """读取 CSV，按 name 分组，每组返回按 t_env 排序的 (t_env, value) 列表。"""
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            series[row["name"]].append((float(row["t_env"]), float(row["value"])))
    for name in series:
        series[name].sort()
    return series


def _to_metric_list(metrics) -> list[str] | None:
    """把 metrics 归一化成 list[str]；None 表示全部。支持 None / 字符串 / 列表。"""
    if metrics is None:
        return None
    if isinstance(metrics, str):
        return [m.strip() for m in metrics.split(",") if m.strip()]
    return [str(m) for m in metrics]


def _plot_one(ax, name, points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.2)
    ax.set_xlabel("t_env")
    ax.grid(True, alpha=0.3)


def draw_curves(series, names, same_plot, cols, title, output):
    if same_plot:
        fig, ax = plt.subplots(figsize=(12, 6))
        for name in names:
            _plot_one(ax, name, series[name])
            ax.get_lines()[-1].set_label(name)
        ax.set_ylabel("value")
        ax.legend(fontsize=8, ncol=2)
        if title:
            ax.set_title(title)
    else:
        n = len(names)
        ncols = max(1, min(cols, n))
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 3.2 * nrows),
            squeeze=False,
        )
        for i, name in enumerate(names):
            ax = axes[i // ncols][i % ncols]
            _plot_one(ax, name, series[name])
            ax.set_ylabel(name, fontsize=8)
            ax.set_title(name, fontsize=9)
        # 隐藏多余的空白子图
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved to {output}")

def main():
    parser = argparse.ArgumentParser(
        description="把训练日志 CSV（t_env,name,value）转成折线图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="/mnt/e/ReinforcementLearning/workdir/config/plot.yaml", help="绘图配置文件（yaml）")
    parser.add_argument("--csv", default=None, help="覆盖 yaml 里的 csv 输入路径")
    parser.add_argument("-o", "--output", default=None, help="覆盖 yaml 里的 output")
    parser.add_argument("--metrics", default=None, help="覆盖 yaml 里的 metrics（逗号分隔）")
    parser.add_argument("--same-plot", action=argparse.BooleanOptionalAction, default=None,
                        help="覆盖 yaml 里的 same_plot")
    parser.add_argument("--cols", type=int, default=None, help="覆盖 yaml 里的 cols")
    parser.add_argument("--title", default=None, help="覆盖 yaml 里的 title")
    cli = parser.parse_args()

    cfg = OmegaConf.load(cli.config)

    # ---- 命令行覆盖（None 表示用 yaml）----
    csv_path = Path(cli.csv if cli.csv is not None else cfg.csv)
    output = Path(cli.output) if cli.output is not None else (
        Path(cfg.output) if cfg.output is not None else csv_path.with_suffix(".png")
    )
    metrics = _to_metric_list(cli.metrics if cli.metrics is not None else cfg.get("metrics"))
    same_plot = cli.same_plot if cli.same_plot is not None else cfg.get("same_plot", False)
    cols = cli.cols if cli.cols is not None else cfg.get("cols", 2)
    title = cli.title if cli.title is not None else cfg.get("title")

    series = load_series(csv_path)
    if not series:
        raise SystemExit(f"{csv_path} 里没有数据")

    if metrics is not None:
        missing = [n for n in metrics if n not in series]
        if missing:
            raise SystemExit(f"CSV 里没有这些指标: {missing}\n可用指标: {', '.join(series)}")
        names = metrics
    else:
        names = list(series)

    print(f"共 {len(names)} 个指标: {', '.join(names)}")
    draw_curves(series, names, same_plot, cols, title, output)


if __name__ == "__main__":
    main()
