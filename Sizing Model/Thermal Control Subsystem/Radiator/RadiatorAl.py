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
class ThermalControlRadiatorAl(om.ExplicitComponent):
    """Radiator sizing"""

    sigma_radiator_Al: float
    alpha_radiator_Al: float
    epsilon_radiator_Al: float
    beta_radiator_Al: float
    SolarConstant_radiator_Al: float
    Q_int_Al: float
    q_EarthIR_Al: float
    T_req_radiator_Al: float
    t_radiator_Al: float
    rho_radiator_Al: float

    def __init__(self, sigma_radiator_Al, alpha_radiator_Al, epsilon_radiator_Al, beta_radiator_Al, SolarConstant_radiator_Al,
                 Q_int_Al, q_EarthIR_Al, T_req_radiator_Al, t_radiator_Al, rho_radiator_Al):
        super().__init__()

        self.sigma_radiator_Al = sigma_radiator_Al
        self.alpha_radiator_Al = alpha_radiator_Al
        self.epsilon_radiator_Al = epsilon_radiator_Al
        self.beta_radiator_Al = beta_radiator_Al
        self.SolarConstant_radiator_Al = SolarConstant_radiator_Al
        self.Q_int_Al = Q_int_Al
        self.q_EarthIR_Al = q_EarthIR_Al
        self.T_req_radiator_Al = T_req_radiator_Al
        self.t_radiator_Al = t_radiator_Al
        self.rho_radiator_Al = rho_radiator_Al

    def setup(self):
        # Inputs
        self.add_input('sigma_radiator_Al', val=self.sigma_radiator_Al)
        self.add_input('alpha_radiator_Al', val=self.alpha_radiator_Al)
        self.add_input('epsilon_radiator_Al', val=self.epsilon_radiator_Al)
        self.add_input('beta_radiator_Al', val=self.beta_radiator_Al)
        self.add_input('SolarConstant_radiator_Al', val=self.SolarConstant_radiator_Al, units="W/m**2")
        self.add_input('Q_int_Al', val=self.Q_int_Al, units="W")
        self.add_input('q_EarthIR_Al', val=self.q_EarthIR_Al, units="W/m**2")
        self.add_input('T_req_radiator_Al', val=self.T_req_radiator_Al, units="K")
        self.add_input('t_radiator_Al', val=self.t_radiator_Al, units="mm")
        self.add_input('rho_radiator_Al', val=self.rho_radiator_Al, units="kg/m**3")

        # Outputs
        self.add_output('A_radiator', units="m**2")
        self.add_output('M_thermal_radiator', units="kg")  # shared variable

        # Every output depends on several inputs
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.sigma_radiator_Al = inputs['sigma_radiator_Al']
        self.alpha_radiator_Al = inputs['alpha_radiator_Al']
        self.epsilon_radiator_Al = inputs['epsilon_radiator_Al']
        self.beta_radiator_Al = inputs['beta_radiator_Al']
        self.SolarConstant_radiator_Al = inputs['SolarConstant_radiator_Al']
        self.Q_int_Al = inputs['Q_int_Al']
        self.q_EarthIR_Al = inputs['q_EarthIR_Al']
        self.T_req_radiator_Al = inputs['T_req_radiator_Al']
        self.t_radiator_Al = inputs['t_radiator_Al']
        self.rho_radiator_Al = inputs['rho_radiator_Al']

        q_ext = self.alpha_radiator_Al * (self.SolarConstant_radiator_Al + self.beta_radiator_Al * self.SolarConstant_radiator_Al +
                                           self.q_EarthIR_Al)
        q_rad = self.epsilon_radiator_Al * self.sigma_radiator_Al * self.T_req_radiator_Al ** 4
        self.A_radiator = self.Q_int_Al / (q_rad - q_ext)
        self.M_thermal_radiator = self.A_radiator * (self.t_radiator_Al * 0.001) * self.rho_radiator_Al

        outputs['A_radiator'] = self.A_radiator
        outputs['M_thermal_radiator'] = self.M_thermal_radiator