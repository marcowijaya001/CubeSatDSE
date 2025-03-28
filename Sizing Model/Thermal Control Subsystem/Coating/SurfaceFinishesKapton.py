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
class ThermalControlSurfaceFinishesKapton(om.ExplicitComponent):
    """Surface finishes/coatings as passive thermal control sizing (Kapton variant)"""

    sigma_coating_kapton: float
    alpha_coating_kapton: float
    epsilon_coating_kapton: float
    SolarConstant_coating_kapton: float
    edge_size_coating_kapton: float
    height_size_coating_kapton: float
    t_coating_kapton: float
    rho_coating_kapton: float
    T_req_coating_kapton: float

    def __init__(self, sigma_coating_kapton, alpha_coating_kapton, epsilon_coating_kapton, SolarConstant_coating_kapton,
                 edge_size_coating_kapton, height_size_coating_kapton, t_coating_kapton, rho_coating_kapton, T_req_coating_kapton):
        super().__init__()

        self.sigma_coating_kapton = sigma_coating_kapton
        self.alpha_coating_kapton = alpha_coating_kapton
        self.epsilon_coating_kapton = epsilon_coating_kapton
        self.SolarConstant_coating_kapton = SolarConstant_coating_kapton
        self.edge_size_coating_kapton = edge_size_coating_kapton
        self.height_size_coating_kapton = height_size_coating_kapton
        self.t_coating_kapton = t_coating_kapton
        self.rho_coating_kapton = rho_coating_kapton
        self.T_req_coating_kapton = T_req_coating_kapton

    def setup(self):
        # Inputs
        self.add_input('sigma_coating_kapton', val=self.sigma_coating_kapton, units="W/m**2/K**4")
        self.add_input('alpha_coating_kapton', val=self.alpha_coating_kapton)
        self.add_input('epsilon_coating_kapton', val=self.epsilon_coating_kapton)
        self.add_input('SolarConstant_coating_kapton', val=self.SolarConstant_coating_kapton, units="W/m**2")
        self.add_input('edge_size_coating_kapton', val=self.edge_size_coating_kapton, units="cm")
        self.add_input('height_size_coating_kapton', val=self.height_size_coating_kapton, units="cm")
        self.add_input('t_coating_kapton', val=self.t_coating_kapton, units="mm")
        self.add_input('rho_coating_kapton', val=self.rho_coating_kapton, units="kg/m**2")
        self.add_input('T_req_coating_kapton', val=self.T_req_coating_kapton, units="K")

        # Outputs
        self.add_output('M_thermal_coating', units="kg")  # shared variable
        self.add_output('A_coating', units="m**2")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.sigma_coating_kapton = inputs['sigma_coating_kapton']
        self.alpha_coating_kapton = inputs['alpha_coating_kapton']
        self.epsilon_coating_kapton = inputs['epsilon_coating_kapton']
        self.SolarConstant_coating_kapton = inputs['SolarConstant_coating_kapton']
        self.edge_size_coating_kapton = inputs['edge_size_coating_kapton']
        self.height_size_coating_kapton = inputs['height_size_coating_kapton']
        self.t_coating_kapton = inputs['t_coating_kapton']
        self.rho_coating_kapton = inputs['rho_coating_kapton']
        self.T_req_coating_kapton = inputs['T_req_coating_kapton']

        A = 4 * self.edge_size_coating_kapton * self.height_size_coating_kapton + 2 * self.edge_size_coating_kapton ** 2

        self.A_coating = (self.sigma_coating_kapton * self.T_req_coating_kapton ** 4 * A / 10000) / (
                    self.alpha_coating_kapton * self.SolarConstant_coating_kapton / self.epsilon_coating_kapton)
        self.M_thermal_coating = self.rho_coating_kapton * self.A_coating * (self.t_coating_kapton * 0.001)

        outputs['M_thermal_coating'] = self.M_thermal_coating
        outputs['A_coating'] = self.A_coating