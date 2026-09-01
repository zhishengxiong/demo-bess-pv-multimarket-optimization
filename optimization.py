"""Build and solve the sequential DA-IMB models and the joint co-optimization model for all forecast-accuracy cases."""

import numpy as np
from gurobipy import GRB, Model, quicksum


# ====================================================================================================
# DA optimization
# ====================================================================================================

def build_da_model(data, day_idx):

    # ====================================================================================================
    # 1. Parameters
    # ====================================================================================================

    T = data["T"]

    load = data["load"][day_idx]
    pv_available = data["pv"][day_idx]
    da_price = data["da_price"][day_idx]

    ess_pmax = data["ess_pmax"]
    ess_capacity = data["ess_capacity"]
    ess_eini = data["ess_eini"]
    ess_eff = data["ess_eff"]

    grid_limit = data["grid_limit"]

    # ====================================================================================================
    # 2. Model
    # ====================================================================================================

    m = Model(f"DA_optimization_day_{day_idx + 1}")
    m.setParam("OutputFlag", 0)

    # ====================================================================================================
    # 3. Decision variables
    # ====================================================================================================

    P_net_DA = m.addMVar(
        T,
        lb=-grid_limit,
        ub=grid_limit,
        vtype=GRB.CONTINUOUS,
        name="P_net_DA",
    )

    PV_DA = m.addMVar(
        T,
        lb=0,
        ub=pv_available,
        vtype=GRB.CONTINUOUS,
        name="PV_DA",
    )

    bess_ch_DA = m.addMVar(
        T,
        lb=0,
        ub=ess_pmax,
        vtype=GRB.CONTINUOUS,
        name="bess_ch_DA",
    )

    bess_dis_DA = m.addMVar(
        T,
        lb=0,
        ub=ess_pmax,
        vtype=GRB.CONTINUOUS,
        name="bess_dis_DA",
    )

    bess_E_DA = m.addMVar(
        T,
        lb=0.2 * ess_capacity,
        ub=0.8 * ess_capacity,
        vtype=GRB.CONTINUOUS,
        name="bess_E_DA",
    )

    bess_u_DA = m.addMVar(
        T,
        vtype=GRB.BINARY,
        name="bess_u_DA",
    )

    # ====================================================================================================
    # 4. Power balance
    # ====================================================================================================

    for t in range(T):

        m.addConstr(
            P_net_DA[t]
            ==
            PV_DA[t]
            + bess_dis_DA[t]
            - bess_ch_DA[t]
            - load[t],
            name=f"power_balance_DA_{t}",
        )

    # ====================================================================================================
    # 5. BESS charging / discharging constraints
    # ====================================================================================================

    for t in range(T):

        m.addConstr(
            bess_ch_DA[t]
            <= ess_pmax * bess_u_DA[t],
            name=f"bess_ch_limit_DA_{t}",
        )

        m.addConstr(
            bess_dis_DA[t]
            <= ess_pmax * (1 - bess_u_DA[t]),
            name=f"bess_dis_limit_DA_{t}",
        )

    # ====================================================================================================
    # 6. BESS energy balance
    # ====================================================================================================

    for t in range(T):

        if t == 0:

            m.addConstr(
                bess_E_DA[t]
                ==
                ess_eini
                + ess_eff * bess_ch_DA[t]
                - bess_dis_DA[t] / ess_eff,
                name=f"bess_energy_DA_{t}",
            )

        else:

            m.addConstr(
                bess_E_DA[t]
                ==
                bess_E_DA[t - 1]
                + ess_eff * bess_ch_DA[t]
                - bess_dis_DA[t] / ess_eff,
                name=f"bess_energy_DA_{t}",
            )

    m.addConstr(
        bess_E_DA[T - 1] == ess_eini,
        name="bess_terminal_energy_DA",
    )

    # ====================================================================================================
    # 7. Objective
    # ====================================================================================================

    da_revenue = quicksum(
        da_price[t] * P_net_DA[t]
        for t in range(T)
    )

    m.setObjective(
        da_revenue,
        GRB.MAXIMIZE,
    )

    # ====================================================================================================
    # 8. Store variables
    # ====================================================================================================

    m._xvars = {
        "P_net_DA": P_net_DA,
        "PV_DA": PV_DA,
        "bess_ch_DA": bess_ch_DA,
        "bess_dis_DA": bess_dis_DA,
        "bess_E_DA": bess_E_DA,
        "bess_u_DA": bess_u_DA,
    }

    return m


def solve_da_model(data, day_idx):

    m = build_da_model(
        data,
        day_idx,
    )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"DA optimization failed for Day {day_idx + 1}. "
            f"Gurobi status: {m.Status}"
        )

    x = m._xvars

    return {
        "day": day_idx + 1,
        "objective": float(m.ObjVal),
        "P_net_DA": np.array(x["P_net_DA"].X),
        "PV_DA": np.array(x["PV_DA"].X),
        "bess_ch_DA": np.array(x["bess_ch_DA"].X),
        "bess_dis_DA": np.array(x["bess_dis_DA"].X),
        "bess_E_DA": np.array(x["bess_E_DA"].X),
        "bess_u_DA": np.array(x["bess_u_DA"].X),
    }


# ====================================================================================================
# IMB optimization
# ====================================================================================================

def build_imb_model(
    data,
    day_idx,
    da_result,
):

    # ====================================================================================================
    # 1. Parameters
    # ====================================================================================================

    T = data["T"]

    load = data["load"][day_idx]
    pv_available = data["pv"][day_idx]
    imb_price = data["imb_price"][day_idx]

    P_net_DA_hat = da_result["P_net_DA"]

    ess_pmax = data["ess_pmax"]
    ess_capacity = data["ess_capacity"]
    ess_eini = data["ess_eini"]
    ess_eff = data["ess_eff"]

    grid_limit = data["grid_limit"]

    # ====================================================================================================
    # 2. Model
    # ====================================================================================================

    m = Model(f"IMB_optimization_day_{day_idx + 1}")
    m.setParam("OutputFlag", 0)

    # ====================================================================================================
    # 3. Decision variables
    # ====================================================================================================

    P_net_IMB = m.addMVar(
        T,
        lb=-grid_limit,
        ub=grid_limit,
        vtype=GRB.CONTINUOUS,
        name="P_net_IMB",
    )

    PV_IMB = m.addMVar(
        T,
        lb=0,
        ub=pv_available,
        vtype=GRB.CONTINUOUS,
        name="PV_IMB",
    )

    bess_ch_IMB = m.addMVar(
        T,
        lb=0,
        ub=ess_pmax,
        vtype=GRB.CONTINUOUS,
        name="bess_ch_IMB",
    )

    bess_dis_IMB = m.addMVar(
        T,
        lb=0,
        ub=ess_pmax,
        vtype=GRB.CONTINUOUS,
        name="bess_dis_IMB",
    )

    bess_E_IMB = m.addMVar(
        T,
        lb=0.2 * ess_capacity,
        ub=0.8 * ess_capacity,
        vtype=GRB.CONTINUOUS,
        name="bess_E_IMB",
    )

    bess_u_IMB = m.addMVar(
        T,
        vtype=GRB.BINARY,
        name="bess_u_IMB",
    )

    # ====================================================================================================
    # 4. Power balance
    # ====================================================================================================

    for t in range(T):

        m.addConstr(
            P_net_IMB[t]
            ==
            PV_IMB[t]
            + bess_dis_IMB[t]
            - bess_ch_IMB[t]
            - load[t],
            name=f"power_balance_IMB_{t}",
        )

    # ====================================================================================================
    # 5. BESS charging / discharging constraints
    # ====================================================================================================

    for t in range(T):

        m.addConstr(
            bess_ch_IMB[t]
            <= ess_pmax * bess_u_IMB[t],
            name=f"bess_ch_limit_IMB_{t}",
        )

        m.addConstr(
            bess_dis_IMB[t]
            <= ess_pmax * (1 - bess_u_IMB[t]),
            name=f"bess_dis_limit_IMB_{t}",
        )

    # ====================================================================================================
    # 6. BESS energy balance
    # ====================================================================================================

    for t in range(T):

        if t == 0:

            m.addConstr(
                bess_E_IMB[t]
                ==
                ess_eini
                + ess_eff * bess_ch_IMB[t]
                - bess_dis_IMB[t] / ess_eff,
                name=f"bess_energy_IMB_{t}",
            )

        else:

            m.addConstr(
                bess_E_IMB[t]
                ==
                bess_E_IMB[t - 1]
                + ess_eff * bess_ch_IMB[t]
                - bess_dis_IMB[t] / ess_eff,
                name=f"bess_energy_IMB_{t}",
            )

    m.addConstr(
        bess_E_IMB[T - 1] == ess_eini,
        name="bess_terminal_energy_IMB",
    )

    # ====================================================================================================
    # 7. Objective
    # ====================================================================================================

    imb_revenue = quicksum(
        imb_price[t]
        * (
            P_net_IMB[t]
            - P_net_DA_hat[t]
        )
        for t in range(T)
    )

    m.setObjective(
        imb_revenue,
        GRB.MAXIMIZE,
    )

    # ====================================================================================================
    # 8. Store variables
    # ====================================================================================================

    m._xvars = {
        "P_net_IMB": P_net_IMB,
        "PV_IMB": PV_IMB,
        "bess_ch_IMB": bess_ch_IMB,
        "bess_dis_IMB": bess_dis_IMB,
        "bess_E_IMB": bess_E_IMB,
        "bess_u_IMB": bess_u_IMB,
    }

    return m


def solve_imb_model(
    data,
    day_idx,
    da_result,
):

    m = build_imb_model(
        data,
        day_idx,
        da_result,
    )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"IMB optimization failed for Day {day_idx + 1}. "
            f"Gurobi status: {m.Status}"
        )

    x = m._xvars

    P_net_IMB = np.array(
        x["P_net_IMB"].X
    )

    P_net_DA_hat = da_result["P_net_DA"]

    imbalance_deviation = (
        P_net_IMB
        - P_net_DA_hat
    )

    return {
        "day": day_idx + 1,
        "objective": float(m.ObjVal),
        "P_net_IMB": P_net_IMB,
        "imbalance_deviation": imbalance_deviation,
        "PV_IMB": np.array(x["PV_IMB"].X),
        "bess_ch_IMB": np.array(x["bess_ch_IMB"].X),
        "bess_dis_IMB": np.array(x["bess_dis_IMB"].X),
        "bess_E_IMB": np.array(x["bess_E_IMB"].X),
        "bess_u_IMB": np.array(x["bess_u_IMB"].X),
    }


# ====================================================================================================
# Co-optimization
# ====================================================================================================

def build_co_optimization_model(
    data,
    day_idx,
):

    # ====================================================================================================
    # 1. Parameters
    # ====================================================================================================

    T = data["T"]

    load = data["load"][day_idx]
    pv_available = data["pv"][day_idx]

    da_price = data["da_price"][day_idx]
    imb_price = data["imb_price"][day_idx]

    ess_pmax = data["ess_pmax"]
    ess_capacity = data["ess_capacity"]
    ess_eini = data["ess_eini"]
    ess_eff = data["ess_eff"]

    grid_limit = data["grid_limit"]

    # ====================================================================================================
    # 2. Model
    # ====================================================================================================

    m = Model(
        f"Co_optimization_day_{day_idx + 1}"
    )

    m.setParam(
        "OutputFlag",
        0,
    )

    # ====================================================================================================
    # 3. DA decision variables
    # ====================================================================================================

    P_net_DA = m.addMVar(
        T,
        lb=-grid_limit,
        ub=grid_limit,
        vtype=GRB.CONTINUOUS,
        name="P_net_DA",
    )

    PV_DA = m.addMVar(
        T,
        lb=0,
        ub=pv_available,
        vtype=GRB.CONTINUOUS,
        name="PV_DA",
    )

    bess_ch_DA = m.addMVar(
        T,
        lb=0,
        ub=ess_pmax,
        vtype=GRB.CONTINUOUS,
        name="bess_ch_DA",
    )

    bess_dis_DA = m.addMVar(
        T,
        lb=0,
        ub=ess_pmax,
        vtype=GRB.CONTINUOUS,
        name="bess_dis_DA",
    )

    bess_E_DA = m.addMVar(
        T,
        lb=0.2 * ess_capacity,
        ub=0.8 * ess_capacity,
        vtype=GRB.CONTINUOUS,
        name="bess_E_DA",
    )

    bess_u_DA = m.addMVar(
        T,
        vtype=GRB.BINARY,
        name="bess_u_DA",
    )

    # ====================================================================================================
    # 4. IMB decision variables
    # ====================================================================================================

    P_net_IMB = m.addMVar(
        T,
        lb=-grid_limit,
        ub=grid_limit,
        vtype=GRB.CONTINUOUS,
        name="P_net_IMB",
    )

    PV_IMB = m.addMVar(
        T,
        lb=0,
        ub=pv_available,
        vtype=GRB.CONTINUOUS,
        name="PV_IMB",
    )

    bess_ch_IMB = m.addMVar(
        T,
        lb=0,
        ub=ess_pmax,
        vtype=GRB.CONTINUOUS,
        name="bess_ch_IMB",
    )

    bess_dis_IMB = m.addMVar(
        T,
        lb=0,
        ub=ess_pmax,
        vtype=GRB.CONTINUOUS,
        name="bess_dis_IMB",
    )

    bess_E_IMB = m.addMVar(
        T,
        lb=0.2 * ess_capacity,
        ub=0.8 * ess_capacity,
        vtype=GRB.CONTINUOUS,
        name="bess_E_IMB",
    )

    bess_u_IMB = m.addMVar(
        T,
        vtype=GRB.BINARY,
        name="bess_u_IMB",
    )

    # ====================================================================================================
    # 5. DA power balance
    # ====================================================================================================

    for t in range(T):

        m.addConstr(
            P_net_DA[t]
            ==
            PV_DA[t]
            + bess_dis_DA[t]
            - bess_ch_DA[t]
            - load[t],
            name=f"power_balance_DA_{t}",
        )

    # ====================================================================================================
    # 6. DA BESS constraints
    # ====================================================================================================

    for t in range(T):

        m.addConstr(
            bess_ch_DA[t]
            <= ess_pmax * bess_u_DA[t],
            name=f"bess_ch_limit_DA_{t}",
        )

        m.addConstr(
            bess_dis_DA[t]
            <= ess_pmax * (1 - bess_u_DA[t]),
            name=f"bess_dis_limit_DA_{t}",
        )

    for t in range(T):

        if t == 0:

            m.addConstr(
                bess_E_DA[t]
                ==
                ess_eini
                + ess_eff * bess_ch_DA[t]
                - bess_dis_DA[t] / ess_eff,
                name=f"bess_energy_DA_{t}",
            )

        else:

            m.addConstr(
                bess_E_DA[t]
                ==
                bess_E_DA[t - 1]
                + ess_eff * bess_ch_DA[t]
                - bess_dis_DA[t] / ess_eff,
                name=f"bess_energy_DA_{t}",
            )

    m.addConstr(
        bess_E_DA[T - 1] == ess_eini,
        name="bess_terminal_energy_DA",
    )

    # ====================================================================================================
    # 7. IMB power balance
    # ====================================================================================================

    for t in range(T):

        m.addConstr(
            P_net_IMB[t]
            ==
            PV_IMB[t]
            + bess_dis_IMB[t]
            - bess_ch_IMB[t]
            - load[t],
            name=f"power_balance_IMB_{t}",
        )

    # ====================================================================================================
    # 8. IMB BESS constraints
    # ====================================================================================================

    for t in range(T):

        m.addConstr(
            bess_ch_IMB[t]
            <= ess_pmax * bess_u_IMB[t],
            name=f"bess_ch_limit_IMB_{t}",
        )

        m.addConstr(
            bess_dis_IMB[t]
            <= ess_pmax * (1 - bess_u_IMB[t]),
            name=f"bess_dis_limit_IMB_{t}",
        )

    for t in range(T):

        if t == 0:

            m.addConstr(
                bess_E_IMB[t]
                ==
                ess_eini
                + ess_eff * bess_ch_IMB[t]
                - bess_dis_IMB[t] / ess_eff,
                name=f"bess_energy_IMB_{t}",
            )

        else:

            m.addConstr(
                bess_E_IMB[t]
                ==
                bess_E_IMB[t - 1]
                + ess_eff * bess_ch_IMB[t]
                - bess_dis_IMB[t] / ess_eff,
                name=f"bess_energy_IMB_{t}",
            )

    m.addConstr(
        bess_E_IMB[T - 1] == ess_eini,
        name="bess_terminal_energy_IMB",
    )

    # ====================================================================================================
    # 9. Objective
    # ====================================================================================================

    total_revenue = quicksum(
        da_price[t] * P_net_DA[t]
        +
        imb_price[t]
        * (
            P_net_IMB[t]
            - P_net_DA[t]
        )
        for t in range(T)
    )

    m.setObjective(
        total_revenue,
        GRB.MAXIMIZE,
    )

    # ====================================================================================================
    # 10. Store variables
    # ====================================================================================================

    m._xvars = {
        "P_net_DA": P_net_DA,
        "PV_DA": PV_DA,
        "bess_ch_DA": bess_ch_DA,
        "bess_dis_DA": bess_dis_DA,
        "bess_E_DA": bess_E_DA,
        "bess_u_DA": bess_u_DA,
        "P_net_IMB": P_net_IMB,
        "PV_IMB": PV_IMB,
        "bess_ch_IMB": bess_ch_IMB,
        "bess_dis_IMB": bess_dis_IMB,
        "bess_E_IMB": bess_E_IMB,
        "bess_u_IMB": bess_u_IMB,
    }

    return m


def solve_co_optimization_model(
    data,
    day_idx,
):

    m = build_co_optimization_model(
        data,
        day_idx,
    )

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Co-optimization failed for Day {day_idx + 1}. "
            f"Gurobi status: {m.Status}"
        )

    x = m._xvars

    P_net_DA = np.array(
        x["P_net_DA"].X
    )

    P_net_IMB = np.array(
        x["P_net_IMB"].X
    )

    imbalance_deviation = (
        P_net_IMB
        - P_net_DA
    )

    da_price = data["da_price"][day_idx]
    imb_price = data["imb_price"][day_idx]

    da_revenue = float(
        np.sum(
            da_price * P_net_DA
        )
    )

    imb_revenue = float(
        np.sum(
            imb_price * imbalance_deviation
        )
    )

    return {
        "day": day_idx + 1,
        "objective": float(m.ObjVal),
        "da_revenue": da_revenue,
        "imb_revenue": imb_revenue,
        "P_net_DA": P_net_DA,
        "P_net_IMB": P_net_IMB,
        "imbalance_deviation": imbalance_deviation,
        "PV_DA": np.array(x["PV_DA"].X),
        "PV_IMB": np.array(x["PV_IMB"].X),
        "bess_ch_DA": np.array(x["bess_ch_DA"].X),
        "bess_dis_DA": np.array(x["bess_dis_DA"].X),
        "bess_E_DA": np.array(x["bess_E_DA"].X),
        "bess_u_DA": np.array(x["bess_u_DA"].X),
        "bess_ch_IMB": np.array(x["bess_ch_IMB"].X),
        "bess_dis_IMB": np.array(x["bess_dis_IMB"].X),
        "bess_E_IMB": np.array(x["bess_E_IMB"].X),
        "bess_u_IMB": np.array(x["bess_u_IMB"].X),
    }

# ====================================================================================================
# Run all forecast-based optimization cases
# ====================================================================================================

def run_all_optimizations(
    forecast_data,
):

    all_results = {}

    for case_name, data in forecast_data.items():

        case_results = []

        for day_idx in range(data["num_days"]):

            # Sequential optimization: DA -> IMB
            da_result = solve_da_model(
                data,
                day_idx,
            )

            imb_result = solve_imb_model(
                data,
                day_idx,
                da_result,
            )

            # Co-optimization: DA and IMB optimized simultaneously
            co_result = solve_co_optimization_model(
                data,
                day_idx,
            )

            case_results.append(
                {
                    "day": day_idx + 1,
                    "da": da_result,
                    "imb": imb_result,
                    "co_optimization": co_result,
                }
            )

        all_results[case_name] = case_results

    return all_results


# ====================================================================================================
# Theoretical optimum under perfect foresight
# ====================================================================================================

def solve_theoretical_optimum(
    actual_data,
):

    daily_results = []

    for day_idx in range(actual_data["num_days"]):

        # Perfect foresight:
        # the co-optimization model directly uses the realized DA and IMB prices.
        result = solve_co_optimization_model(
            actual_data,
            day_idx,
        )

        daily_results.append(
            result
        )

    theoretical_total = sum(
        result["objective"]
        for result in daily_results
    )

    return {
        "daily": daily_results,
        "total": theoretical_total,
    }
