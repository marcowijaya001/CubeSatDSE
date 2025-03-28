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
class PrimaryBatteryAgZn(om.ExplicitComponent):
    """Primary battery sizing"""

    MD_AgZn: float
    e_pb_AgZn: float

    def __init__(self, MD_AgZn, e_pb_AgZn):
        super().__init__()

        self.MD_AgZn = MD_AgZn
        self.e_pb_AgZn = e_pb_AgZn

    def setup(self):
        # Inputs (with _AgZn for MD and e_pb; P_total remains unchanged)
        self.add_input('MD_AgZn', val=self.MD_AgZn)
        self.add_input('P_total', units="W")  # shared variable (unchanged)
        self.add_input('e_pb_AgZn', val=self.e_pb_AgZn)

        # Outputs (unchanged)
        self.add_output('E_battery_pb')
        self.add_output('Mpb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MD_AgZn = inputs['MD_AgZn']
        P_total = inputs['P_total']  # unchanged
        e_pb_AgZn = inputs['e_pb_AgZn']

        # Primary battery calculation
        Margin = 0.2
        E_battery_pb = (1 + Margin) * P_total * MD_AgZn * 24
        Mpb = E_battery_pb / e_pb_AgZn

        # Assign outputs
        outputs['E_battery_pb'] = E_battery_pb
        outputs['Mpb'] = Mpb