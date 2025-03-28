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
class SecondaryBattery(om.ExplicitComponent):
    """Secondary battery sizing"""

    MaxET: float
    DoD: float
    Ed: float

    # E_battery_sb: float
    # Msb: float

    def __init__(self, MaxET, DoD, Ed):
        super().__init__()

        self.MaxET = MaxET
        self.DoD = DoD
        self.Ed = Ed
        # self.E_battery_sb = E_battery_sb
        # self.Msb = Msb

    def setup(self):
        self.add_input('MaxET', val=self.MaxET, units="min")
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('DoD', val=self.DoD)
        self.add_input('Ed', val=self.Ed)

        self.add_output('E_battery_sb')
        self.add_output('Msb', units="kg")  # shared variable

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.MaxET = inputs['MaxET']
        P_total = inputs['P_total']
        self.DoD = inputs['DoD']
        self.Ed = inputs['Ed']

        Margin = 0.5
        E_eclipse = (1 + Margin) * P_total * self.MaxET * 60 / 3600
        self.E_battery_sb = E_eclipse / self.DoD
        self.Msb = self.E_battery_sb / self.Ed

        outputs['E_battery_sb'] = self.E_battery_sb
        outputs['Msb'] = self.Msb