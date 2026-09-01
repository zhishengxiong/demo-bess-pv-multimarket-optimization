"""Evaluate optimized market decisions against same-day actual prices and summarize realized revenues."""

import numpy as np


def calculate_realized_revenue(
    P_net_DA,
    P_net_IMB,
    da_actual,
    imb_actual,
):
    da_revenue = np.sum(
        da_actual * P_net_DA
    )

    imb_revenue = np.sum(
        imb_actual * (P_net_IMB - P_net_DA)
    )

    return float(
        da_revenue + imb_revenue
    )


def backtest_case(
    optimization_results,
    actual_data,
):
    sequential_daily = []
    co_optimization_daily = []

    for day_idx, result in enumerate(optimization_results):

        sequential_revenue = calculate_realized_revenue(
            P_net_DA=result["da"]["P_net_DA"],
            P_net_IMB=result["imb"]["P_net_IMB"],
            da_actual=actual_data["da_price"][day_idx],
            imb_actual=actual_data["imb_price"][day_idx],
        )

        co_optimization_revenue = calculate_realized_revenue(
            P_net_DA=result["co_optimization"]["P_net_DA"],
            P_net_IMB=result["co_optimization"]["P_net_IMB"],
            da_actual=actual_data["da_price"][day_idx],
            imb_actual=actual_data["imb_price"][day_idx],
        )

        sequential_daily.append(
            sequential_revenue
        )

        co_optimization_daily.append(
            co_optimization_revenue
        )

        difference = (
            co_optimization_revenue
            - sequential_revenue
        )

        print(
            f"Day {day_idx + 1:2d}: "
            f"Sequential optimization = {sequential_revenue:10.2f} | "
            f"Co-optimization = {co_optimization_revenue:10.2f} | "
            f"Difference = {difference:10.2f}"
        )

    return {
        "sequential_daily": sequential_daily,
        "co_optimization_daily": co_optimization_daily,
        "sequential_total": sum(sequential_daily),
        "co_optimization_total": sum(co_optimization_daily),
    }


def run_backtest(
    optimization_results,
    actual_data,
):
    all_results = {}

    for case_name, case_results in optimization_results.items():

        print()
        print("-" * 100)
        print(f"{case_name} forecast accuracy")
        print("-" * 100)

        all_results[case_name] = backtest_case(
            case_results,
            actual_data,
        )

    return all_results


def add_capture_rates(
    backtest_results,
    theoretical_total,
):
    for result in backtest_results.values():

        result["sequential_capture_rate"] = (
            result["sequential_total"]
            / theoretical_total
            * 100
        )

        result["co_optimization_capture_rate"] = (
            result["co_optimization_total"]
            / theoretical_total
            * 100
        )

    return backtest_results


def print_summary(
    all_results,
    theoretical_total,
):
    print()
    print("=" * 120)
    print("Final realized-revenue comparison")
    print("=" * 120)

    print(
        f"Theoretical optimum (perfect-foresight co-optimization) = "
        f"{theoretical_total:.2f}"
    )

    print("-" * 120)

    for accuracy_level in ["Low", "Medium", "High"]:

        result = all_results[accuracy_level]

        difference = (
            result["co_optimization_total"]
            - result["sequential_total"]
        )

        print(
            f"{accuracy_level:6s} forecast accuracy: "
            f"Sequential = {result['sequential_total']:10.2f} "
            f"({result['sequential_capture_rate']:7.2f}% capture) | "
            f"Co-opt = {result['co_optimization_total']:10.2f} "
            f"({result['co_optimization_capture_rate']:7.2f}% capture) | "
            f"Difference = {difference:10.2f}"
        )

    print("=" * 120)
