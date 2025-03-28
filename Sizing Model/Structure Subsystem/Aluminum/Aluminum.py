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
class StructureAluminum(om.ExplicitComponent):
    """Structural sizing of CubeSat using aluminum as material"""

    g_aluminum: float
    rho_aluminum: float
    tau_aluminum: float
    safety_factor_aluminum: float
    width_aluminum: float
    height_size_aluminum: float
    edge_size_aluminum: float

    # t_structure_aluminum: float
    # M_structure_aluminum: float

    def __init__(self, g_aluminum, rho_aluminum, tau_aluminum, safety_factor_aluminum,
                 width_aluminum, height_size_aluminum, edge_size_aluminum):
        super().__init__()

        self.g_aluminum = g_aluminum
        self.rho_aluminum = rho_aluminum
        self.tau_aluminum = tau_aluminum
        self.safety_factor_aluminum = safety_factor_aluminum
        self.width_aluminum = width_aluminum
        self.height_size_aluminum = height_size_aluminum
        self.edge_size_aluminum = edge_size_aluminum
        # self.t_structure_aluminum = t_structure_aluminum
        # self.M_structure_aluminum = M_structure_aluminum

    def setup(self):
        # Inputs
        self.add_input('M_total', units="kg")  # shared variable
        self.add_input('g_aluminum', val=self.g_aluminum, units="m/s**2")
        self.add_input('rho_aluminum', val=self.rho_aluminum, units="kg/m**3")
        self.add_input('tau_aluminum', val=self.tau_aluminum, units="N/m**2")
        self.add_input('safety_factor_aluminum', val=self.safety_factor_aluminum)
        self.add_input('width_aluminum', val=self.width_aluminum, units="cm")
        self.add_input('height_size_aluminum', val=self.height_size_aluminum, units="cm")
        self.add_input('edge_size_aluminum', val=self.edge_size_aluminum, units="cm")

        # Outputs
        self.add_output('t_structure_aluminum', units="mm")
        self.add_output('M_structure_aluminum', units="kg")  # shared variable

        # Every output depends on several inputs
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        M_total = inputs['M_total']
        self.g_aluminum = inputs['g_aluminum']
        self.rho_aluminum = inputs['rho_aluminum']
        self.tau_aluminum = inputs['tau_aluminum']
        self.safety_factor_aluminum = inputs['safety_factor_aluminum']
        self.width_aluminum = inputs['width_aluminum']
        self.height_size_aluminum = inputs['height_size_aluminum']
        self.edge_size_aluminum = inputs['edge_size_aluminum']

        F = M_total * self.g_aluminum * self.safety_factor_aluminum
        P = self.width_aluminum * 0.01
        self.t_structure_aluminum = F / (P * self.tau_aluminum)
        A_structure = (4 * (self.edge_size_aluminum * 0.01) * (self.height_size_aluminum * 0.01) +
                       2 * (self.edge_size_aluminum * 0.01) ** 2) * 0.5
        self.M_structure_aluminum = self.rho_aluminum * A_structure * self.t_structure_aluminum

        outputs['t_structure_aluminum'] = self.t_structure_aluminum
        outputs['M_structure_aluminum'] = self.M_structure_aluminum