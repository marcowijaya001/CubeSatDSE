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
class Magnetorquers(om.ExplicitComponent):
    """Sizing of magnetic torquer"""

    T_total_mag: float
    B_mag: float
    I_mag: float
    A_mag: float
    r_mag: float
    rho_mag: float
    d_mag: float
    rho_wire: float

    def __init__(self, T_total_mag, B_mag, I_mag, A_mag, r_mag, rho_mag, d_mag, rho_wire):
        super().__init__()

        self.T_total_mag = T_total_mag
        self.B_mag = B_mag
        self.I_mag = I_mag
        self.A_mag = A_mag
        self.r_mag = r_mag
        self.rho_mag = rho_mag
        self.d_mag = d_mag
        self.rho_wire = rho_wire

    def setup(self):
        self.add_input('T_total_mag', val=self.T_total_mag, units="N*m")
        self.add_input('B_mag', val=self.B_mag, units="T")
        self.add_input('I_mag', val=self.I_mag, units="A")
        self.add_input('A_mag', val=self.A_mag, units="m**2")
        self.add_input('r_mag', val=self.r_mag, units="m")
        self.add_input('rho_mag', val=self.rho_mag)
        self.add_input('d_mag', val=self.d_mag, units="m")
        self.add_input('rho_wire', val=self.rho_wire, units="kg/m**3")

        self.add_output('D_mag', units="A*m**2")
        self.add_output('N_mag')
        self.add_output('L_mag', units="m")
        self.add_output('R_mag')
        self.add_output('P_mag', units="W")
        self.add_output('m_mag', units="kg")

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        self.T_total_mag = inputs['T_total_mag']
        self.B_mag = inputs['B_mag']
        self.I_mag = inputs['I_mag']
        self.A_mag = inputs['A_mag']
        self.r_mag = inputs['r_mag']
        self.rho_mag = inputs['rho_mag']
        self.d_mag = inputs['d_mag']
        self.rho_wire = inputs['rho_wire']

        self.D_mag = self.T_total_mag / self.B_mag
        self.N_mag = self.D_mag / (self.I_mag * self.A_mag)
        self.L_mag = 2 * math.pi * self.r_mag * self.N_mag
        Ac = math.pi * (self.d_mag / 2) ** 2
        self.R_mag = self.rho_mag * self.L_mag / Ac
        self.P_mag = self.I_mag ** 2 * self.R_mag
        self.m_mag = self.rho_wire * self.L_mag * Ac

        outputs['D_mag'] = self.D_mag
        outputs['N_mag'] = self.N_mag
        outputs['L_mag'] = self.L_mag
        outputs['R_mag'] = self.R_mag
        outputs['P_mag'] = self.P_mag
        outputs['m_mag'] = self.m_mag