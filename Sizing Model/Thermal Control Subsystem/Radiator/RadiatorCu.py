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
class ThermalControlRadiatorCu(om.ExplicitComponent):
    """Radiator sizing"""

    sigma_radiator_Cu: float
    alpha_radiator_Cu: float
    epsilon_radiator_Cu: float
    beta_radiator_Cu: float
    SolarConstant_radiator_Cu: float
    Q_int_Cu: float
    q_EarthIR_Cu: float
    T_req_radiator_Cu: float
    t_radiator_Cu: float
    rho_radiator_Cu: float

    def __init__(self, sigma_radiator_Cu, alpha_radiator_Cu, epsilon_radiator_Cu, beta_radiator_Cu, SolarConstant_radiator_Cu,
                 Q_int_Cu, q_EarthIR_Cu, T_req_radiator_Cu, t_radiator_Cu, rho_radiator_Cu):
        super().__init__()

        self.sigma_radiator_Cu = sigma_radiator_Cu
        self.alpha_radiator_Cu = alpha_radiator_Cu
        self.epsilon_radiator_Cu = epsilon_radiator_Cu
        self.beta_radiator_Cu = beta_radiator_Cu
        self.SolarConstant_radiator_Cu = SolarConstant_radiator_Cu
        self.Q_int_Cu = Q_int_Cu
        self.q_EarthIR_Cu = q_EarthIR_Cu
        self.T_req_radiator_Cu = T_req_radiator_Cu
        self.t_radiator_Cu = t_radiator_Cu
        self.rho_radiator_Cu = rho_radiator_Cu

    def setup(self):
        # Inputs
        self.add_input('sigma_radiator_Cu', val=self.sigma_radiator_Cu)
        self.add_input('alpha_radiator_Cu', val=self.alpha_radiator_Cu)
        self.add_input('epsilon_radiator_Cu', val=self.epsilon_radiator_Cu)
        self.add_input('beta_radiator_Cu', val=self.beta_radiator_Cu)
        self.add_input('SolarConstant_radiator_Cu', val=self.SolarConstant_radiator_Cu, units="W/m**2")
        self.add_input('Q_int_Cu', val=self.Q_int_Cu, units="W")
        self.add_input('q_EarthIR_Cu', val=self.q_EarthIR_Cu, units="W/m**2")
        self.add_input('T_req_radiator_Cu', val=self.T_req_radiator_Cu, units="K")
        self.add_input('t_radiator_Cu', val=self.t_radiator_Cu, units="mm")
        self.add_input('rho_radiator_Cu', val=self.rho_radiator_Cu, units="kg/m**3")

        # Outputs
        self.add_output('A_radiator', units="m**2")
        self.add_output('M_thermal_radiator', units="kg")  # shared variable

        # Every output depends on several inputs
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.sigma_radiator_Cu = inputs['sigma_radiator_Cu']
        self.alpha_radiator_Cu = inputs['alpha_radiator_Cu']
        self.epsilon_radiator_Cu = inputs['epsilon_radiator_Cu']
        self.beta_radiator_Cu = inputs['beta_radiator_Cu']
        self.SolarConstant_radiator_Cu = inputs['SolarConstant_radiator_Cu']
        self.Q_int_Cu = inputs['Q_int_Cu']
        self.q_EarthIR_Cu = inputs['q_EarthIR_Cu']
        self.T_req_radiator_Cu = inputs['T_req_radiator_Cu']
        self.t_radiator_Cu = inputs['t_radiator_Cu']
        self.rho_radiator_Cu = inputs['rho_radiator_Cu']

        q_ext = self.alpha_radiator_Cu * (self.SolarConstant_radiator_Cu + self.beta_radiator_Cu * self.SolarConstant_radiator_Cu +
                                           self.q_EarthIR_Cu)
        q_rad = self.epsilon_radiator_Cu * self.sigma_radiator_Cu * self.T_req_radiator_Cu ** 4
        self.A_radiator = self.Q_int_Cu / (q_rad - q_ext)
        self.M_thermal_radiator = self.A_radiator * (self.t_radiator_Cu * 0.001) * self.rho_radiator_Cu

        outputs['A_radiator'] = self.A_radiator
        outputs['M_thermal_radiator'] = self.M_thermal_radiator