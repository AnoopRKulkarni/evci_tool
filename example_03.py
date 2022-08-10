# AUTOGENERATED! DO NOT EDIT! File to edit: ../index.ipynb.

# %% auto 0
__all__ = []

# %% ../index.ipynb 1
from evci_tool.config import *
from evci_tool.model import *
from evci_tool.analysis import *

ui_inputs = { 
    "M": ["3WS", "4WS"],
    "years_of_analysis": 2,
    "capex_2W": 2500,
    "capex_3WS": 112000,
    "capex_4WS": 250000,
    "capex_4WF": 1500000,
    "hoarding cost": 900000,
    "kiosk_cost": 180000,
    "year1_conversion": 0.02,
    "year2_conversion": 0.05,
    "year3_conversion": 0.1,
    "holiday_percentage": 0.3,
    "fast_charging": 0.3,
    "slow_charging": 0.15,
}

# %% ../index.ipynb 7
if __name__ == "__main__":
    analyze_sites ('chandigarh_leh', ui_inputs, cluster=True, use_defaults=True)