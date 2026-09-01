"""Run the complete workflow: data preprocessing, optimization, backtesting, and visualization."""

from pathlib import Path

from backtest import (
    add_capture_rates,
    print_summary,
    run_backtest,
)
from data_preprocessing import load_all_data
from optimization import (
    run_all_optimizations,
    solve_theoretical_optimum,
)
from visualization import plot_all_results


def main():

    project_dir = Path(__file__).resolve().parent

    # ====================================================================================================
    # 1. Data preprocessing
    # ====================================================================================================

    data = load_all_data(
        project_dir / "data"
    )

    # ====================================================================================================
    # 2. Optimization
    # ====================================================================================================

    optimization_results = run_all_optimizations(
        data["forecast"]
    )

    theoretical_optimum = solve_theoretical_optimum(
        data["actual"]
    )

    # ====================================================================================================
    # 3. Backtest
    # ====================================================================================================

    backtest_results = run_backtest(
        optimization_results,
        data["actual"],
    )

    backtest_results = add_capture_rates(
        backtest_results,
        theoretical_optimum["total"],
    )

    print_summary(
        backtest_results,
        theoretical_optimum["total"],
    )

    # ====================================================================================================
    # 4. Visualization
    # ====================================================================================================

    plot_all_results(
        backtest_results,
        theoretical_optimum,
        project_dir / "figures",
    )


if __name__ == "__main__":
    main()