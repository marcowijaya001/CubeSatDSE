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
class ThermalControlSurfaceFinishesWhite(om.ExplicitComponent):
    """Surface finishes/coatings as passive thermal control sizing (White variant)"""

    sigma_coating_white: float
    alpha_coating_white: float
    epsilon_coating_white: float
    SolarConstant_coating_white: float
    edge_size_coating_white: float
    height_size_coating_white: float
    t_coating_white: float
    rho_coating_white: float
    T_req_coating_white: float

    def __init__(self, sigma_coating_white, alpha_coating_white, epsilon_coating_white, SolarConstant_coating_white,
                 edge_size_coating_white, height_size_coating_white, t_coating_white, rho_coating_white, T_req_coating_white):
        super().__init__()

        self.sigma_coating_white = sigma_coating_white
        self.alpha_coating_white = alpha_coating_white
        self.epsilon_coating_white = epsilon_coating_white
        self.SolarConstant_coating_white = SolarConstant_coating_white
        self.edge_size_coating_white = edge_size_coating_white
        self.height_size_coating_white = height_size_coating_white
        self.t_coating_white = t_coating_white
        self.rho_coating_white = rho_coating_white
        self.T_req_coating_white = T_req_coating_white

    def setup(self):
        # Inputs
        self.add_input('sigma_coating_white', val=self.sigma_coating_white, units="W/m**2/K**4")
        self.add_input('alpha_coating_white', val=self.alpha_coating_white)
        self.add_input('epsilon_coating_white', val=self.epsilon_coating_white)
        self.add_input('SolarConstant_coating_white', val=self.SolarConstant_coating_white, units="W/m**2")
        self.add_input('edge_size_coating_white', val=self.edge_size_coating_white, units="cm")
        self.add_input('height_size_coating_white', val=self.height_size_coating_white, units="cm")
        self.add_input('t_coating_white', val=self.t_coating_white, units="mm")
        self.add_input('rho_coating_white', val=self.rho_coating_white, units="kg/m**2")
        self.add_input('T_req_coating_white', val=self.T_req_coating_white, units="K")

        # Outputs
        self.add_output('M_thermal_coating', units="kg")  # shared variable
        self.add_output('A_coating', units="m**2")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.sigma_coating_white = inputs['sigma_coating_white']
        self.alpha_coating_white = inputs['alpha_coating_white']
        self.epsilon_coating_white = inputs['epsilon_coating_white']
        self.SolarConstant_coating_white = inputs['SolarConstant_coating_white']
        self.edge_size_coating_white = inputs['edge_size_coating_white']
        self.height_size_coating_white = inputs['height_size_coating_white']
        self.t_coating_white = inputs['t_coating_white']
        self.rho_coating_white = inputs['rho_coating_white']
        self.T_req_coating_white = inputs['T_req_coating_white']

        A = 4 * self.edge_size_coating_white * self.height_size_coating_white + 2 * self.edge_size_coating_white ** 2

        self.A_coating = (self.sigma_coating_white * self.T_req_coating_white ** 4 * A / 10000) / (
                    self.alpha_coating_white * self.SolarConstant_coating_white / self.epsilon_coating_white)
        self.M_thermal_coating = self.rho_coating_white * self.A_coating * (self.t_coating_white * 0.001)

        outputs['M_thermal_coating'] = self.M_thermal_coating
        outputs['A_coating'] = self.A_coating