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
class PrimaryBatteryLiCl(om.ExplicitComponent):
    """Primary battery sizing"""

    MD_LiCl: float
    e_pb_LiCl: float

    def __init__(self, MD_LiCl, e_pb_LiCl):
        super().__init__()

        self.MD_LiCl = MD_LiCl
        self.e_pb_LiCl = e_pb_LiCl

    def setup(self):
        # Inputs (with _LiCl for MD and e_pb; P_total remains unchanged)
        self.add_input('MD_LiCl', val=self.MD_LiCl)
        self.add_input('P_total', units="W")  # shared variable (unchanged)
        self.add_input('e_pb_LiCl', val=self.e_pb_LiCl)

        # Outputs (unchanged)
        self.add_output('E_battery_pb')
        self.add_output('Mpb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MD_LiCl = inputs['MD_LiCl']
        P_total = inputs['P_total']  # unchanged
        e_pb_LiCl = inputs['e_pb_LiCl']

        # Primary battery calculation
        Margin = 0.2
        E_battery_pb = (1 + Margin) * P_total * MD_LiCl * 24
        Mpb = E_battery_pb / e_pb_LiCl

        # Assign outputs
        outputs['E_battery_pb'] = E_battery_pb
        outputs['Mpb'] = Mpb