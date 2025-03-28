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
class ThermalControlSurfaceFinishes(om.ExplicitComponent):
    """Surface finishes/coatings as passive thermal control sizing"""

    sigma_coating: float
    alpha_coating: float
    epsilon_coating: float
    SolarConstant_coating: float
    edge_size_coating: float
    height_size_coating: float
    t_coating: float
    rho_coating: float
    T_req_coating: float

    # M_thermal_coating: float
    # A_coating: float

    def __init__(self, sigma_coating, alpha_coating, epsilon_coating, SolarConstant_coating, edge_size_coating,
                 height_size_coating, t_coating, rho_coating, T_req_coating):
        super().__init__()

        self.sigma_coating = sigma_coating
        self.alpha_coating = alpha_coating
        self.epsilon_coating = epsilon_coating
        self.SolarConstant_coating = SolarConstant_coating
        self.edge_size_coating = edge_size_coating
        self.height_size_coating = height_size_coating
        self.t_coating = t_coating
        self.rho_coating = rho_coating
        self.T_req_coating = T_req_coating
        # self.M_thermal_coating = M_thermal_coating
        # self.A_coating = A_coating

    def setup(self):
        # Inputs
        self.add_input('sigma_coating', val=self.sigma_coating, units="W/m**2/K**4")
        self.add_input('alpha_coating', val=self.alpha_coating)
        self.add_input('epsilon_coating', val=self.epsilon_coating)
        self.add_input('SolarConstant_coating', val=self.SolarConstant_coating, units="W/m**2")
        self.add_input('edge_size_coating', val=self.edge_size_coating, units="cm")
        self.add_input('height_size_coating', val=self.height_size_coating, units="cm")
        self.add_input('t_coating', val=self.t_coating, units="mm")
        self.add_input('rho_coating', val=self.rho_coating, units="kg/m**2")
        self.add_input('T_req_coating', val=self.T_req_coating, units="K")

        # Outputs
        # self.add_output('T_operating', units="C")
        self.add_output('M_thermal_coating', units="kg")  # shared variable
        self.add_output('A_coating', units="m**2")

        # Every output depends on several inputs
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.sigma_coating = inputs['sigma_coating']
        self.alpha_coating = inputs['alpha_coating']
        self.epsilon_coating = inputs['epsilon_coating']
        self.SolarConstant_coating = inputs['SolarConstant_coating']
        self.edge_size_coating = inputs['edge_size_coating']
        self.height_size_coating = inputs['height_size_coating']
        self.t_coating = inputs['t_coating']
        self.rho_coating = inputs['rho_coating']
        self.T_req_coating = inputs['T_req_coating']

        A = 4 * self.edge_size_coating * self.height_size_coating + 2 * self.edge_size_coating ** 2

        self.A_coating = (self.sigma_coating * self.T_req_coating ** 4 * A / 10000) / (
                    self.alpha_coating * self.SolarConstant_coating / self.epsilon_coating)
        self.M_thermal_coating = self.rho_coating * self.A_coating * (self.t_coating * 0.001)

        outputs['M_thermal_coating'] = self.M_thermal_coating
        outputs['A_coating'] = self.A_coating