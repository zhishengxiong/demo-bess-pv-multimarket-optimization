"""Create summary and revenue figures comparing sequential optimization and co-optimization."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ACCURACY_LEVELS = [
    "Low",
    "Medium",
    "High",
]


def format_k(value, _):
    return f"{value / 1000:g}k"


# ====================================================================================================
# Figure 1: Overall expected revenue
# ====================================================================================================

def plot_total_revenue(
    all_results,
    theoretical_optimum,
    output_dir,
):
    accuracy_labels = [
        "Low forecast accuracy",
        "Medium forecast accuracy",
        "High forecast accuracy",
    ]

    sequential = [
        all_results[level]["sequential_total"]
        for level in ACCURACY_LEVELS
    ]

    co_optimization = [
        all_results[level]["co_optimization_total"]
        for level in ACCURACY_LEVELS
    ]

    x = np.arange(
        len(accuracy_labels)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        x - width / 2,
        sequential,
        width,
        label="Sequential optimization",
    )

    ax.bar(
        x + width / 2,
        co_optimization,
        width,
        label="Co-optimization",
    )

    # Theoretical optimum under perfect foresight
    ax.axhline(
        y=theoretical_optimum["total"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Theoretical optimum",
    )

    # Zero line
    ax.axhline(
        y=0,
        color="black",
        linewidth=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        accuracy_labels
    )

    ax.set_ylabel(
        "Expected revenue"
    )

    ax.set_title(
        "Expected Revenue in One Month: Sequential Optimization vs Co-optimization"
    )

    # Revenue axis in k
    ax.yaxis.set_major_formatter(
        FuncFormatter(format_k)
    )

    # Fix upper limit at 300k
    lower_limit = min(
        min(sequential),
        min(co_optimization),
    ) * 1.05

    ax.set_ylim(
        bottom=lower_limit,
        top=300000,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "expected_revenue.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ====================================================================================================
# Figures 2-4: Daily revenue
# ====================================================================================================

def plot_daily_revenue(
    result,
    theoretical_optimum,
    accuracy_level,
    output_dir,
):
    days = np.arange(
        1,
        len(result["sequential_daily"]) + 1,
    )

    theoretical_daily = [
        day_result["objective"]
        for day_result in theoretical_optimum["daily"]
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        days,
        result["sequential_daily"],
        label="Sequential optimization",
        linewidth=2,
    )

    ax.plot(
        days,
        result["co_optimization_daily"],
        label="Co-optimization",
        linewidth=2,
    )

    # Theoretical optimum under perfect foresight
    ax.plot(
        days,
        theoretical_daily,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Theoretical optimum",
    )

    ax.set_xlabel(
        "Day"
    )

    ax.set_ylabel(
        "Revenue"
    )

    ax.set_title(
        f"Daily Revenue over One Month - {accuracy_level} Forecast Accuracy"
    )

    # Revenue axis in k
    ax.yaxis.set_major_formatter(
        FuncFormatter(format_k)
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / f"daily_revenue_{accuracy_level.lower()}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# ====================================================================================================
# Plot all figures
# ====================================================================================================

def plot_all_results(
    all_results,
    theoretical_optimum,
    output_dir,
):
    output_dir.mkdir(
        exist_ok=True
    )

    # Figure 1
    plot_total_revenue(
        all_results,
        theoretical_optimum,
        output_dir,
    )

    # Figures 2-4
    for accuracy_level in ACCURACY_LEVELS:

        plot_daily_revenue(
            all_results[accuracy_level],
            theoretical_optimum,
            accuracy_level,
            output_dir,
        )