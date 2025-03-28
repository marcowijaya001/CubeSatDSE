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
class ThermalControlSurfaceFinishesBlack(om.ExplicitComponent):
    """Surface finishes/coatings as passive thermal control sizing (Black variant)"""

    sigma_coating_black: float
    alpha_coating_black: float
    epsilon_coating_black: float
    SolarConstant_coating_black: float
    edge_size_coating_black: float
    height_size_coating_black: float
    t_coating_black: float
    rho_coating_black: float
    T_req_coating_black: float

    def __init__(self, sigma_coating_black, alpha_coating_black, epsilon_coating_black, SolarConstant_coating_black,
                 edge_size_coating_black, height_size_coating_black, t_coating_black, rho_coating_black, T_req_coating_black):
        super().__init__()

        self.sigma_coating_black = sigma_coating_black
        self.alpha_coating_black = alpha_coating_black
        self.epsilon_coating_black = epsilon_coating_black
        self.SolarConstant_coating_black = SolarConstant_coating_black
        self.edge_size_coating_black = edge_size_coating_black
        self.height_size_coating_black = height_size_coating_black
        self.t_coating_black = t_coating_black
        self.rho_coating_black = rho_coating_black
        self.T_req_coating_black = T_req_coating_black

    def setup(self):
        # Inputs
        self.add_input('sigma_coating_black', val=self.sigma_coating_black, units="W/m**2/K**4")
        self.add_input('alpha_coating_black', val=self.alpha_coating_black)
        self.add_input('epsilon_coating_black', val=self.epsilon_coating_black)
        self.add_input('SolarConstant_coating_black', val=self.SolarConstant_coating_black, units="W/m**2")
        self.add_input('edge_size_coating_black', val=self.edge_size_coating_black, units="cm")
        self.add_input('height_size_coating_black', val=self.height_size_coating_black, units="cm")
        self.add_input('t_coating_black', val=self.t_coating_black, units="mm")
        self.add_input('rho_coating_black', val=self.rho_coating_black, units="kg/m**2")
        self.add_input('T_req_coating_black', val=self.T_req_coating_black, units="K")

        # Outputs
        self.add_output('M_thermal_coating', units="kg")  # shared variable
        self.add_output('A_coating', units="m**2")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.sigma_coating_black = inputs['sigma_coating_black']
        self.alpha_coating_black = inputs['alpha_coating_black']
        self.epsilon_coating_black = inputs['epsilon_coating_black']
        self.SolarConstant_coating_black = inputs['SolarConstant_coating_black']
        self.edge_size_coating_black = inputs['edge_size_coating_black']
        self.height_size_coating_black = inputs['height_size_coating_black']
        self.t_coating_black = inputs['t_coating_black']
        self.rho_coating_black = inputs['rho_coating_black']
        self.T_req_coating_black = inputs['T_req_coating_black']

        A = 4 * self.edge_size_coating_black * self.height_size_coating_black + 2 * self.edge_size_coating_black ** 2

        self.A_coating = (self.sigma_coating_black * self.T_req_coating_black ** 4 * A / 10000) / (
                    self.alpha_coating_black * self.SolarConstant_coating_black / self.epsilon_coating_black)
        self.M_thermal_coating = self.rho_coating_black * self.A_coating * (self.t_coating_black * 0.001)

        outputs['M_thermal_coating'] = self.M_thermal_coating
        outputs['A_coating'] = self.A_coating