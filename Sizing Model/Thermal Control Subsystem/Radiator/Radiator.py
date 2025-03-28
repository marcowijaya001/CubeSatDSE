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
class ThermalControlRadiator(om.ExplicitComponent):
    """Radiator sizing"""

    sigma_radiator: float
    alpha_radiator: float
    epsilon_radiator: float
    beta_radiator: float
    SolarConstant_radiator: float
    Q_int: float
    q_EarthIR: float
    T_req_radiator: float
    t_radiator: float
    rho_radiator: float

    # M_thermal_radiator: float
    # A_radiator: float

    def __init__(self, sigma_radiator, alpha_radiator, epsilon_radiator, beta_radiator, SolarConstant_radiator,
                 Q_int, q_EarthIR, T_req_radiator, t_radiator, rho_radiator):
        super().__init__()

        self.sigma_radiator = sigma_radiator
        self.alpha_radiator = alpha_radiator
        self.epsilon_radiator = epsilon_radiator
        self.beta_radiator = beta_radiator
        self.SolarConstant_radiator = SolarConstant_radiator
        self.Q_int = Q_int
        self.q_EarthIR = q_EarthIR
        self.T_req_radiator = T_req_radiator
        self.t_radiator = t_radiator
        self.rho_radiator = rho_radiator
        # self.M_thermal_radiator = M_thermal_radiator
        # self.A_radiator = A_radiator

    def setup(self):
        # Inputs
        self.add_input('sigma_radiator', val=self.sigma_radiator)
        self.add_input('alpha_radiator', val=self.alpha_radiator)
        self.add_input('epsilon_radiator', val=self.epsilon_radiator)
        self.add_input('beta_radiator', val=self.beta_radiator)
        self.add_input('SolarConstant_radiator', val=self.SolarConstant_radiator, units="W/m**2")
        self.add_input('Q_int', val=self.Q_int, units="W")
        self.add_input('q_EarthIR', val=self.q_EarthIR, units="W/m**2")
        self.add_input('T_req_radiator', val=self.T_req_radiator, units="K")
        self.add_input('t_radiator', val=self.t_radiator, units="mm")
        self.add_input('rho_radiator', val=self.rho_radiator, units="kg/m**3")

        # Outputs
        self.add_output('A_radiator', units="m**2")
        self.add_output('M_thermal_radiator', units="kg")  # shared variable

        # Every output depends on several inputs
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.sigma_radiator = inputs['sigma_radiator']
        self.alpha_radiator = inputs['alpha_radiator']
        self.epsilon_radiator = inputs['epsilon_radiator']
        self.beta_radiator = inputs['beta_radiator']
        self.SolarConstant_radiator = inputs['SolarConstant_radiator']
        self.Q_int = inputs['Q_int']
        self.q_EarthIR = inputs['q_EarthIR']
        self.T_req_radiator = inputs['T_req_radiator']
        self.t_radiator = inputs['t_radiator']
        self.rho_radiator = inputs['rho_radiator']

        q_ext = self.alpha_radiator * (self.SolarConstant_radiator + self.beta_radiator * self.SolarConstant_radiator +
                                       self.q_EarthIR)
        q_rad = self.epsilon_radiator * self.sigma_radiator * self.T_req_radiator ** 4
        self.A_radiator = self.Q_int / (q_rad - q_ext)
        self.M_thermal_radiator = self.A_radiator * (self.t_radiator * 0.001) * self.rho_radiator

        outputs['A_radiator'] = self.A_radiator
        outputs['M_thermal_radiator'] = self.M_thermal_radiator