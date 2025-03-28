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
class ReactionWheel(om.ExplicitComponent):
    """Sizing of reaction wheel"""

    Rwr: float
    Rwav: float
    Op_rw: float
    t_slew: float
    MaxSA: float
    edge_size_rw: float
    height_size_rw: float

    # h_size_rw: float
    # Mrw: float
    # Prw: float

    def __init__(self, Rwr, Rwav, Op_rw, t_slew, MaxSA, edge_size_rw, height_size_rw):
        super().__init__()

        self.Rwr = Rwr
        self.Rwav = Rwav
        self.Op_rw = Op_rw
        self.t_slew = t_slew
        self.MaxSA = MaxSA
        self.edge_size_rw = edge_size_rw
        self.height_size_rw = height_size_rw
        # self.h_size_rw = h_size_rw
        # self.Mrw = Mrw
        # self.Prw = Prw

    def setup(self):
        self.add_input('M_total', units="kg")  # shared variable
        self.add_input('Rwr', val=self.Rwr, units="m")
        self.add_input('Rwav', val=self.Rwav, units="rad/s")
        self.add_input('Op_rw', val=self.Op_rw, units="min")
        self.add_input('t_slew', val=self.t_slew, units="s")
        self.add_input('MaxSA', val=self.MaxSA, units="deg")
        self.add_input('edge_size_rw', val=self.edge_size_rw, units="cm")
        self.add_input('height_size_rw', val=self.height_size_rw, units="cm")

        self.add_output('h_size_rw', units="N*m*s")
        self.add_output('Mrw', units="kg")  # shared variable
        self.add_output('Prw', units="W")  # shared variable

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        M_total = inputs['M_total']
        self.Rwr = inputs['Rwr']
        self.Rwav = inputs['Rwav']
        self.Op_rw = inputs['Op_rw']
        self.t_slew = inputs['t_slew']
        self.MaxSA = inputs['MaxSA']
        self.edge_size_rw = inputs['edge_size_rw']
        self.height_size_rw = inputs['height_size_rw']

        Icubesat = 1 / 12 * M_total * ((self.edge_size_rw * 0.01) ** 2 + (self.height_size_rw * 0.01) ** 2)
        T_slew = 4 * (self.MaxSA * math.pi / 180) * Icubesat / (self.t_slew ** 2)
        h_slew = T_slew * self.t_slew
        self.h_size_rw = 3 * h_slew
        self.Mrw = self.h_size_rw / (self.Rwav * self.Rwr ** 2)

        alpha = 0.01
        Torque = Icubesat * alpha
        Pmech = Torque * self.Rwav
        MotorEfficiency = 0.8
        self.Prw = Pmech / MotorEfficiency

        outputs['h_size_rw'] = self.h_size_rw
        outputs['Mrw'] = self.Mrw
        outputs['Prw'] = self.Prw