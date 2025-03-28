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
class ThermalControlRadiatorCFRP(om.ExplicitComponent):
    """Radiator sizing"""

    sigma_radiator_CFRP: float
    alpha_radiator_CFRP: float
    epsilon_radiator_CFRP: float
    beta_radiator_CFRP: float
    SolarConstant_radiator_CFRP: float
    Q_int_CFRP: float
    q_EarthIR_CFRP: float
    T_req_radiator_CFRP: float
    t_radiator_CFRP: float
    rho_radiator_CFRP: float

    def __init__(self, sigma_radiator_CFRP, alpha_radiator_CFRP, epsilon_radiator_CFRP, beta_radiator_CFRP, SolarConstant_radiator_CFRP,
                 Q_int_CFRP, q_EarthIR_CFRP, T_req_radiator_CFRP, t_radiator_CFRP, rho_radiator_CFRP):
        super().__init__()

        self.sigma_radiator_CFRP = sigma_radiator_CFRP
        self.alpha_radiator_CFRP = alpha_radiator_CFRP
        self.epsilon_radiator_CFRP = epsilon_radiator_CFRP
        self.beta_radiator_CFRP = beta_radiator_CFRP
        self.SolarConstant_radiator_CFRP = SolarConstant_radiator_CFRP
        self.Q_int_CFRP = Q_int_CFRP
        self.q_EarthIR_CFRP = q_EarthIR_CFRP
        self.T_req_radiator_CFRP = T_req_radiator_CFRP
        self.t_radiator_CFRP = t_radiator_CFRP
        self.rho_radiator_CFRP = rho_radiator_CFRP

    def setup(self):
        # Inputs
        self.add_input('sigma_radiator_CFRP', val=self.sigma_radiator_CFRP)
        self.add_input('alpha_radiator_CFRP', val=self.alpha_radiator_CFRP)
        self.add_input('epsilon_radiator_CFRP', val=self.epsilon_radiator_CFRP)
        self.add_input('beta_radiator_CFRP', val=self.beta_radiator_CFRP)
        self.add_input('SolarConstant_radiator_CFRP', val=self.SolarConstant_radiator_CFRP, units="W/m**2")
        self.add_input('Q_int_CFRP', val=self.Q_int_CFRP, units="W")
        self.add_input('q_EarthIR_CFRP', val=self.q_EarthIR_CFRP, units="W/m**2")
        self.add_input('T_req_radiator_CFRP', val=self.T_req_radiator_CFRP, units="K")
        self.add_input('t_radiator_CFRP', val=self.t_radiator_CFRP, units="mm")
        self.add_input('rho_radiator_CFRP', val=self.rho_radiator_CFRP, units="kg/m**3")

        # Outputs
        self.add_output('A_radiator', units="m**2")
        self.add_output('M_thermal_radiator', units="kg")  # shared variable

        # Every output depends on several inputs
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.sigma_radiator_CFRP = inputs['sigma_radiator_CFRP']
        self.alpha_radiator_CFRP = inputs['alpha_radiator_CFRP']
        self.epsilon_radiator_CFRP = inputs['epsilon_radiator_CFRP']
        self.beta_radiator_CFRP = inputs['beta_radiator_CFRP']
        self.SolarConstant_radiator_CFRP = inputs['SolarConstant_radiator_CFRP']
        self.Q_int_CFRP = inputs['Q_int_CFRP']
        self.q_EarthIR_CFRP = inputs['q_EarthIR_CFRP']
        self.T_req_radiator_CFRP = inputs['T_req_radiator_CFRP']
        self.t_radiator_CFRP = inputs['t_radiator_CFRP']
        self.rho_radiator_CFRP = inputs['rho_radiator_CFRP']

        q_ext = self.alpha_radiator_CFRP * (self.SolarConstant_radiator_CFRP + self.beta_radiator_CFRP * self.SolarConstant_radiator_CFRP +
                                           self.q_EarthIR_CFRP)
        q_rad = self.epsilon_radiator_CFRP * self.sigma_radiator_CFRP * self.T_req_radiator_CFRP ** 4
        self.A_radiator = self.Q_int_CFRP / (q_rad - q_ext)
        self.M_thermal_radiator = self.A_radiator * (self.t_radiator_CFRP * 0.001) * self.rho_radiator_CFRP

        outputs['A_radiator'] = self.A_radiator
        outputs['M_thermal_radiator'] = self.M_thermal_radiator