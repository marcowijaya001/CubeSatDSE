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
class SecondaryBatteryNa2S(om.ExplicitComponent):
    """Secondary battery sizing"""

    MaxET_Na2S: float
    DoD_Na2S: float
    Ed_Na2S: float

    def __init__(self, MaxET_Na2S, DoD_Na2S, Ed_Na2S):
        super().__init__()

        self.MaxET_Na2S = MaxET_Na2S
        self.DoD_Na2S = DoD_Na2S
        self.Ed_Na2S = Ed_Na2S

    def setup(self):
        # Inputs (adding _Na2S suffix)
        self.add_input('MaxET_Na2S', val=self.MaxET_Na2S, units="min")
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('DoD_Na2S', val=self.DoD_Na2S)
        self.add_input('Ed_Na2S', val=self.Ed_Na2S)

        # Outputs (unchanged)
        self.add_output('E_battery_sb')
        self.add_output('Msb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MaxET_Na2S = inputs['MaxET_Na2S']
        P_total = inputs['P_total']
        DoD_Na2S = inputs['DoD_Na2S']
        Ed_Na2S = inputs['Ed_Na2S']

        # Perform calculations
        Margin = 0.5
        E_eclipse = (1 + Margin) * P_total * MaxET_Na2S * 60 / 3600
        E_battery_sb = E_eclipse / DoD_Na2S
        Msb = E_battery_sb / Ed_Na2S

        # Assign outputs
        outputs['E_battery_sb'] = E_battery_sb
        outputs['Msb'] = Msb