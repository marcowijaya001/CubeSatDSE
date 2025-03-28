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
class PrimaryBatteryLiSOCl2(om.ExplicitComponent):
    """Primary battery sizing"""

    MD_LiSOCl2: float
    e_pb_LiSOCl2: float

    def __init__(self, MD_LiSOCl2, e_pb_LiSOCl2):
        super().__init__()

        self.MD_LiSOCl2 = MD_LiSOCl2
        self.e_pb_LiSOCl2 = e_pb_LiSOCl2

    def setup(self):
        # Inputs (with _LiSOCl2 for MD and e_pb; P_total remains unchanged)
        self.add_input('MD_LiSOCl2', val=self.MD_LiSOCl2)
        self.add_input('P_total', units="W")  # shared variable (unchanged)
        self.add_input('e_pb_LiSOCl2', val=self.e_pb_LiSOCl2)

        # Outputs (unchanged)
        self.add_output('E_battery_pb')
        self.add_output('Mpb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MD_LiSOCl2 = inputs['MD_LiSOCl2']
        P_total = inputs['P_total']  # unchanged
        e_pb_LiSOCl2 = inputs['e_pb_LiSOCl2']

        # Primary battery calculation
        Margin = 0.2
        E_battery_pb = (1 + Margin) * P_total * MD_LiSOCl2 * 24
        Mpb = E_battery_pb / e_pb_LiSOCl2

        # Assign outputs
        outputs['E_battery_pb'] = E_battery_pb
        outputs['Mpb'] = Mpb