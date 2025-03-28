from dataclasses import dataclass

import pandas as pd
from adore.optimization.api.factory_evaluator import *
from adore.api.schema import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import openmdao.api as om
import math
import numpy as np
import openmdao.func_api as omf

@dataclass
class SecondaryBatteryLiIon(om.ExplicitComponent):
    """Secondary battery sizing"""

    MaxET_LiIon: float
    DoD_LiIon: float
    Ed_LiIon: float

    def __init__(self, MaxET_LiIon, DoD_LiIon, Ed_LiIon):
        super().__init__()

        self.MaxET_LiIon = MaxET_LiIon
        self.DoD_LiIon = DoD_LiIon
        self.Ed_LiIon = Ed_LiIon

    def setup(self):
        # Inputs (adding _LiIon suffix)
        self.add_input('MaxET_LiIon', val=self.MaxET_LiIon, units="min")
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('DoD_LiIon', val=self.DoD_LiIon)
        self.add_input('Ed_LiIon', val=self.Ed_LiIon)

        # Outputs (unchanged)
        self.add_output('E_battery_sb')
        self.add_output('Msb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MaxET_LiIon = inputs['MaxET_LiIon']
        P_total = inputs['P_total']
        DoD_LiIon = inputs['DoD_LiIon']
        Ed_LiIon = inputs['Ed_LiIon']

        # Perform calculations
        Margin = 0.5
        E_eclipse = (1 + Margin) * P_total * MaxET_LiIon * 60 / 3600
        E_battery_sb = E_eclipse / DoD_LiIon
        Msb = E_battery_sb / Ed_LiIon

        # Assign outputs
        outputs['E_battery_sb'] = E_battery_sb
        outputs['Msb'] = Msb