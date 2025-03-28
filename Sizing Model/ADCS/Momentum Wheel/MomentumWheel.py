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
class MomentumWheel(om.ExplicitComponent):
    """Sizing of momentum wheel"""

    Op_mw: float
    T_total: float
    Mwr: float
    Mwav: float
    Maad: float
    edge_size_mw: float
    height_size_mw: float

    # Mram: float
    # Mmw: float
    # Pmw: float

    def __init__(self, Op_mw, T_total, Mwr, Mwav, Maad, edge_size_mw, height_size_mw):
        super().__init__()

        self.Op_mw = Op_mw
        self.T_total = T_total
        self.Mwr = Mwr
        self.Mwav = Mwav
        self.Maad = Maad
        self.edge_size_mw = edge_size_mw
        self.height_size_mw = height_size_mw
        # self.Mram = Mram
        # self.Mmw = Mmw
        # self.Pmw = Pmw

    def setup(self):
        self.add_input('Op_mw', val=self.Op_mw, units="min")
        self.add_input('T_total', val=self.T_total, units="N*m")
        self.add_input('Mwr', val=self.Mwr, units="m")
        self.add_input('Mwav', val=self.Mwav, units="rad/s")
        self.add_input('Maad', val=self.Maad, units="deg")
        self.add_input('M_total', units="kg")  # shared variable
        self.add_input('edge_size_mw', val=self.edge_size_mw, units="cm")
        self.add_input('height_size_mw', val=self.height_size_mw, units="cm")

        self.add_output('Mram', units="N*m*s")
        self.add_output('Mmw', units="kg")  # shared variable
        self.add_output('Pmw', units="W")  # shared variable

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        self.Op_mw = inputs['Op_mw']
        self.T_total = inputs['T_total']
        self.Mwr = inputs['Mwr']
        self.Mwav = inputs['Mwav']
        self.Maad = inputs['Maad']
        M_total = inputs['M_total']
        self.edge_size_mw = inputs['edge_size_mw']
        self.height_size_mw = inputs['height_size_mw']

        self.Mram = self.T_total * self.Op_mw * 60 / (4 * self.Maad * (math.pi / 180))
        self.Mmw = self.Mram / (self.Mwav * self.Mwr ** 2)

        Icubesat = 1 / 12 * M_total * ((self.edge_size_mw * 0.01) ** 2 + (self.height_size_mw * 0.01) ** 2)
        alpha = 0.01
        Torque = Icubesat * alpha
        Pmech = Torque * self.Mwav
        MotorEfficiency = 0.8
        self.Pmw = Pmech / MotorEfficiency

        outputs['Mram'] = self.Mram
        outputs['Mmw'] = self.Mmw
        outputs['Pmw'] = self.Pmw