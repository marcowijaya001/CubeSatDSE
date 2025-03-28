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
class StructureAdditiveManufacturing(om.ExplicitComponent):
    """Structural sizing of CubeSat using additive manufacturing's materials"""

    g_am: float
    rho_am: float
    tau_am: float
    safety_factor_am: float
    edge_size_am: float
    height_size_am: float
    width_am: float

    # t_structure_am: float
    # M_structure_am: float

    def __init__(self, g_am, rho_am, tau_am, safety_factor_am, edge_size_am, height_size_am, width_am):
        super().__init__()

        self.g_am = g_am
        self.rho_am = rho_am
        self.tau_am = tau_am
        self.safety_factor_am = safety_factor_am
        self.edge_size_am = edge_size_am
        self.height_size_am = height_size_am
        self.width_am = width_am
        # self.t_structure_am = t_structure_am
        # self.M_structure_am = M_structure_am

    def setup(self):
        # Inputs
        self.add_input('M_total', units="kg")  # shared variable
        self.add_input('g_am', val=self.g_am, units="m/s**2")
        self.add_input('rho_am', val=self.rho_am, units="kg/m**3")
        self.add_input('tau_am', val=self.tau_am, units="N/m**2")
        self.add_input('safety_factor_am', val=self.safety_factor_am)
        self.add_input('edge_size_am', val=self.edge_size_am, units="cm")
        self.add_input('height_size_am', val=self.height_size_am, units="cm")
        self.add_input('width_am', val=self.width_am, units="cm")

        # Outputs
        self.add_output('t_structure_am', units="mm")
        self.add_output('M_structure_am', units="kg")  # shared variable

        # Every output depends on several inputs
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        M_total = inputs['M_total']
        self.g_am = inputs['g_am']
        self.rho_am = inputs['rho_am']
        self.tau_am = inputs['tau_am']
        self.safety_factor_am = inputs['safety_factor_am']
        self.width_am = inputs['width_am']
        self.height_size_am = inputs['height_size_am']
        self.edge_size_am = inputs['edge_size_am']

        F = M_total * self.g_am * self.safety_factor_am
        P = self.width_am * 0.01
        self.t_structure_am = F / (P * self.tau_am)
        A_structure = (4 * (self.edge_size_am * 0.01) * (self.height_size_am * 0.01) + 2 * (
                    self.edge_size_am * 0.01) ** 2) * 0.5
        self.M_structure_am = self.rho_am * A_structure * self.t_structure_am

        outputs['t_structure_am'] = self.t_structure_am
        outputs['M_structure_am'] = self.M_structure_am