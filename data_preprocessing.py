"""Load and organize DER data, forecast prices, and actual prices for the optimization and backtest workflow."""

from pathlib import Path

import pandas as pd


T = 24
NUM_DAYS = 30

PRICE_CASES = {
    "Low": "Electricity_Price_Pred_Very_Low_Accuracy.xlsx",
    "Medium": "Electricity_Price_Pred_Low_Accuracy.xlsx",
    "High": "Electricity_Price_Pred_High_Accuracy.xlsx",
}


def load_market_data(data_dir, price_file_name):
    data_dir = Path(data_dir)

    ders_file = data_dir / "DERs_Data.xlsx"
    price_file = data_dir / price_file_name

    ess = pd.read_excel(ders_file, sheet_name="ESS")
    parm = pd.read_excel(ders_file, sheet_name="Parm")

    pv = pd.read_excel(ders_file, sheet_name="PV")
    load = pd.read_excel(ders_file, sheet_name="Load")

    da = pd.read_excel(price_file, sheet_name="DA")
    imb = pd.read_excel(price_file, sheet_name="IMB")

    return {
        "T": T,
        "num_days": NUM_DAYS,
        "grid_limit": float(parm.loc[0, "connection limits"]),
        "ess_pmax": float(ess.loc[0, "Power"]),
        "ess_capacity": float(ess.loc[0, "Energy"]),
        "ess_eini": float(ess.loc[0, "Eini"]),
        "ess_eff": float(ess.loc[0, "Eff"]),
        "pv": pv.iloc[:T, 1:NUM_DAYS + 1].to_numpy(dtype=float).T,
        "load": load.iloc[:T, 1:NUM_DAYS + 1].to_numpy(dtype=float).T,
        "da_price": da.iloc[:T, 1:NUM_DAYS + 1].to_numpy(dtype=float).T,
        "imb_price": imb.iloc[:T, 1:NUM_DAYS + 1].to_numpy(dtype=float).T,
    }


def load_all_data(data_dir):
    data_dir = Path(data_dir)

    forecast_data = {
        case_name: load_market_data(
            data_dir,
            price_file_name,
        )
        for case_name, price_file_name in PRICE_CASES.items()
    }

    actual_data = load_market_data(
        data_dir,
        "Electricity_Price_Actual.xlsx",
    )

    return {
        "forecast": forecast_data,
        "actual": actual_data,
    }
