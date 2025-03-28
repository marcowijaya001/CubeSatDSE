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
class SecondaryBatteryNiCd(om.ExplicitComponent):
    """Secondary battery sizing"""

    MaxET_NiCd: float
    DoD_NiCd: float
    Ed_NiCd: float

    def __init__(self, MaxET_NiCd, DoD_NiCd, Ed_NiCd):
        super().__init__()

        self.MaxET_NiCd = MaxET_NiCd
        self.DoD_NiCd = DoD_NiCd
        self.Ed_NiCd = Ed_NiCd

    def setup(self):
        # Inputs (adding _NiCd suffix)
        self.add_input('MaxET_NiCd', val=self.MaxET_NiCd, units="min")
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('DoD_NiCd', val=self.DoD_NiCd)
        self.add_input('Ed_NiCd', val=self.Ed_NiCd)

        # Outputs (unchanged)
        self.add_output('E_battery_sb')
        self.add_output('Msb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MaxET_NiCd = inputs['MaxET_NiCd']
        P_total = inputs['P_total']
        DoD_NiCd = inputs['DoD_NiCd']
        Ed_NiCd = inputs['Ed_NiCd']

        # Perform calculations
        Margin = 0.5
        E_eclipse = (1 + Margin) * P_total * MaxET_NiCd * 60 / 3600
        E_battery_sb = E_eclipse / DoD_NiCd
        Msb = E_battery_sb / Ed_NiCd

        # Assign outputs
        outputs['E_battery_sb'] = E_battery_sb
        outputs['Msb'] = Msb