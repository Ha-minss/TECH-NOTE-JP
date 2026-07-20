from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager


def _configure_style() -> None:
    font_files = [
        Path(r"C:\Windows\Fonts\malgun.ttf"),
        Path(r"C:\Windows\Fonts\malgunbd.ttf"),
        Path(r"C:\Windows\Fonts\NanumSquareR.ttf"),
    ]
    for font_file in font_files:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
    selected_font = "Malgun Gothic"
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"]:
        if candidate in fonts:
            selected_font = candidate
            break
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "white", "figure.facecolor": "white"})
    plt.rcParams["font.family"] = selected_font
    plt.rcParams["font.sans-serif"] = [selected_font]
    plt.rcParams["axes.unicode_minus"] = False


def _save(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.8)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(summary: pd.DataFrame, output_base: Path) -> None:
    _configure_style()
    metrics = [
        ("arpu", "ARPU", "??? ??", "{:.2f}"),
        ("conversion_rate", "???", "?? ?? ?? ? ??? ??", "{:.3%}"),
        ("arppu", "ARPPU", "???? ??", "{:.0f}"),
    ]
    colors = {"a": "#4C78A8", "b": "#F58518"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (metric, title, ylabel, fmt) in zip(axes, metrics):
        plot_df = summary.reset_index()[["testgroup", metric]]
        sns.barplot(data=plot_df, x="testgroup", y=metric, hue="testgroup", palette=colors, ax=ax, legend=False)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("??")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, plot_df[metric].max() * 1.22)
        for container in ax.containers:
            ax.bar_label(container, labels=[fmt.format(v) for v in container.datavalues], fontsize=9, padding=3)
    fig.suptitle("A/B ?? ?? ??", fontsize=15, fontweight="bold", y=1.03)
    _save(fig, output_base)



def plot_concentration(distribution: pd.DataFrame, output_base: Path) -> None:
    _configure_style()
    cols = [
        "top_1pct_payer_revenue_share",
        "top_5pct_payer_revenue_share",
        "top_10pct_payer_revenue_share",
    ]
    label_map = {
        "top_1pct_payer_revenue_share": "?? 1%",
        "top_5pct_payer_revenue_share": "?? 5%",
        "top_10pct_payer_revenue_share": "?? 10%",
    }
    plot_df = distribution.reset_index().melt(id_vars="testgroup", value_vars=cols, var_name="segment", value_name="share")
    plot_df["segment"] = plot_df["segment"].map(label_map)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    sns.barplot(data=plot_df, x="segment", y="share", hue="testgroup", palette={"a": "#4C78A8", "b": "#F58518"}, ax=ax)
    ax.set_title("??? ?? ??? ?? ???", fontsize=14, fontweight="bold")
    ax.set_xlabel("??? ??")
    ax.set_ylabel("?? ?? ???")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_ylim(0, min(1.0, plot_df["share"].max() * 1.16))
    ax.legend(title="??")
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v:.1%}" for v in container.datavalues], fontsize=9, padding=3)
    _save(fig, output_base)


def plot_bootstrap_distribution(samples, observed: float, ci_low: float, ci_high: float, output_base: Path) -> None:
    _configure_style()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sns.histplot(samples, bins=60, color="#4C78A8", edgecolor="white", ax=ax)
    ax.axvline(observed, color="#111111", linewidth=1.8, label=f"?? ?? {observed:.4f}")
    ax.axvline(ci_low, color="#B23A48", linestyle="--", linewidth=1.6, label=f"95% CI ?? {ci_low:.4f}")
    ax.axvline(ci_high, color="#B23A48", linestyle="--", linewidth=1.6, label=f"95% CI ?? {ci_high:.4f}")
    ax.axvline(0, color="#666666", linestyle=":", linewidth=1.8, label="0 ???")
    ax.set_title("B-A ARPU ??? Bootstrap ??", fontsize=14, fontweight="bold")
    ax.set_xlabel("B-A ARPU ??")
    ax.set_ylabel("Bootstrap ?? ?")
    ax.legend(loc="upper right", frameon=True)
    ax.text(0.02, 0.95, "95% CI? 0? ??", transform=ax.transAxes, fontsize=10, va="top", bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "boxstyle": "round,pad=0.35"})
    _save(fig, output_base)


def plot_sensitivity(sensitivity: pd.DataFrame, output_base: Path) -> None:
    _configure_style()
    label_map = {
        "raw_all_users": "?? ?? ???",
        "common_top_0.1pct_winsorized": "?? ?? 0.1% winsorization",
        "common_top_0.5pct_winsorized": "?? ?? 0.5% winsorization",
    }
    plot_df = sensitivity.copy()
    plot_df["label"] = plot_df["scenario"].map(label_map).fillna(plot_df["scenario"])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = range(len(plot_df))
    ax.errorbar(
        plot_df["b_minus_a"],
        y,
        xerr=[plot_df["b_minus_a"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["b_minus_a"]],
        fmt="o",
        color="#111111",
        ecolor="#4C78A8",
        elinewidth=2,
        capsize=4,
    )
    ax.axvline(0, color="#666666", linestyle=":", linewidth=1.6)
    ax.set_yticks(list(y), plot_df["label"])
    ax.set_title("Whale ???: B-A ARPU ??", fontsize=14, fontweight="bold")
    ax.set_xlabel("B-A ARPU ??? 95% CI")
    ax.set_ylabel("?? ??")
    for yi, value in zip(y, plot_df["b_minus_a"]):
        ax.text(value, yi + 0.13, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    _save(fig, output_base)


def plot_retention_heatmap(heatmap_data: pd.DataFrame, output_base: Path) -> None:
    _configure_style()
    pivot = heatmap_data.pivot(index="cohort_month", columns="day", values="retention_rate")
    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.22)))
    sns.heatmap(pivot, cmap="Blues", annot=False, cbar_kws={"label": "???"}, ax=ax)
    ax.set_title("?? ?? ??? Exact-Day Retention")
    ax.set_xlabel("Retention Day")
    ax.set_ylabel("?? ?")
    _save(fig, output_base)
