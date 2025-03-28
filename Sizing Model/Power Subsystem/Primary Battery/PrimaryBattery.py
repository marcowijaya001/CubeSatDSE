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
class PrimaryBattery(om.ExplicitComponent):
    """Primary battery sizing"""

    MD: float
    e_pb: float

    def __init__(self, MD, e_pb):
        super().__init__()

        self.MD = MD
        self.e_pb = e_pb

    def setup(self):
        self.add_input('MD', val=self.MD)
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('e_pb', val=self.e_pb)

        self.add_output('E_battery_pb')
        self.add_output('Mpb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.MD = inputs['MD']
        P_total = inputs['P_total']
        self.e_pb = inputs['e_pb']

        Margin = 0.2
        self.E_battery_pb = (1 + Margin) * P_total * self.MD * 24
        self.Mpb = self.E_battery_pb / self.e_pb

        outputs['E_battery_pb'] = self.E_battery_pb
        outputs['Mpb'] = self.Mpb
