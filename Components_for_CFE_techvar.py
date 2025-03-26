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

"""
MDO Components in Class Factory Evaluator (CFE) format 
"""

"""
1. Thermal Control Subsystem: Surface Finishes/Coating and Radiator
"""

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


"""
2. Structure Subsystem: Aluminum and Additive Manufacturing
"""

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


"""
3. Attitude Determination and Control Subsystem (ADCS): Reaction Wheel and Momentum Wheel
"""

@dataclass
class ReactionWheel(om.ExplicitComponent):
    """Sizing of reaction wheel"""

    Rwr: float
    Rwav: float
    Op_rw: float
    t_slew: float
    MaxSA: float
    edge_size_rw: float
    height_size_rw: float

    # h_size_rw: float
    # Mrw: float
    # Prw: float

    def __init__(self, Rwr, Rwav, Op_rw, t_slew, MaxSA, edge_size_rw, height_size_rw):
        super().__init__()

        self.Rwr = Rwr
        self.Rwav = Rwav
        self.Op_rw = Op_rw
        self.t_slew = t_slew
        self.MaxSA = MaxSA
        self.edge_size_rw = edge_size_rw
        self.height_size_rw = height_size_rw
        # self.h_size_rw = h_size_rw
        # self.Mrw = Mrw
        # self.Prw = Prw

    def setup(self):
        self.add_input('M_total', units="kg")  # shared variable
        self.add_input('Rwr', val=self.Rwr, units="m")
        self.add_input('Rwav', val=self.Rwav, units="rad/s")
        self.add_input('Op_rw', val=self.Op_rw, units="min")
        self.add_input('t_slew', val=self.t_slew, units="s")
        self.add_input('MaxSA', val=self.MaxSA, units="deg")
        self.add_input('edge_size_rw', val=self.edge_size_rw, units="cm")
        self.add_input('height_size_rw', val=self.height_size_rw, units="cm")

        self.add_output('h_size_rw', units="N*m*s")
        self.add_output('Mrw', units="kg")  # shared variable
        self.add_output('Prw', units="W")  # shared variable

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        M_total = inputs['M_total']
        self.Rwr = inputs['Rwr']
        self.Rwav = inputs['Rwav']
        self.Op_rw = inputs['Op_rw']
        self.t_slew = inputs['t_slew']
        self.MaxSA = inputs['MaxSA']
        self.edge_size_rw = inputs['edge_size_rw']
        self.height_size_rw = inputs['height_size_rw']

        Icubesat = 1 / 12 * M_total * ((self.edge_size_rw * 0.01) ** 2 + (self.height_size_rw * 0.01) ** 2)
        T_slew = 4 * (self.MaxSA * math.pi / 180) * Icubesat / (self.t_slew ** 2)
        h_slew = T_slew * self.t_slew
        self.h_size_rw = 3 * h_slew
        self.Mrw = self.h_size_rw / (self.Rwav * self.Rwr ** 2)

        alpha = 0.01
        Torque = Icubesat * alpha
        Pmech = Torque * self.Rwav
        MotorEfficiency = 0.8
        self.Prw = Pmech / MotorEfficiency

        outputs['h_size_rw'] = self.h_size_rw
        outputs['Mrw'] = self.Mrw
        outputs['Prw'] = self.Prw

@dataclass
class MomentumWheel(om.ExplicitComponent):
    """Sizing of momentum wheel"""

    Op_mw: float
    T_total: float
    Mwr: float
    Mwav: float
    Maad: float
    edge_size_mw: float
    height_size_mw: float

    # Mram: float
    # Mmw: float
    # Pmw: float

    def __init__(self, Op_mw, T_total, Mwr, Mwav, Maad, edge_size_mw, height_size_mw):
        super().__init__()

        self.Op_mw = Op_mw
        self.T_total = T_total
        self.Mwr = Mwr
        self.Mwav = Mwav
        self.Maad = Maad
        self.edge_size_mw = edge_size_mw
        self.height_size_mw = height_size_mw
        # self.Mram = Mram
        # self.Mmw = Mmw
        # self.Pmw = Pmw

    def setup(self):
        self.add_input('Op_mw', val=self.Op_mw, units="min")
        self.add_input('T_total', val=self.T_total, units="N*m")
        self.add_input('Mwr', val=self.Mwr, units="m")
        self.add_input('Mwav', val=self.Mwav, units="rad/s")
        self.add_input('Maad', val=self.Maad, units="deg")
        self.add_input('M_total', units="kg")  # shared variable
        self.add_input('edge_size_mw', val=self.edge_size_mw, units="cm")
        self.add_input('height_size_mw', val=self.height_size_mw, units="cm")

        self.add_output('Mram', units="N*m*s")
        self.add_output('Mmw', units="kg")  # shared variable
        self.add_output('Pmw', units="W")  # shared variable

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        self.Op_mw = inputs['Op_mw']
        self.T_total = inputs['T_total']
        self.Mwr = inputs['Mwr']
        self.Mwav = inputs['Mwav']
        self.Maad = inputs['Maad']
        M_total = inputs['M_total']
        self.edge_size_mw = inputs['edge_size_mw']
        self.height_size_mw = inputs['height_size_mw']

        self.Mram = self.T_total * self.Op_mw * 60 / (4 * self.Maad * (math.pi / 180))
        self.Mmw = self.Mram / (self.Mwav * self.Mwr ** 2)

        Icubesat = 1 / 12 * M_total * ((self.edge_size_mw * 0.01) ** 2 + (self.height_size_mw * 0.01) ** 2)
        alpha = 0.01
        Torque = Icubesat * alpha
        Pmech = Torque * self.Mwav
        MotorEfficiency = 0.8
        self.Pmw = Pmech / MotorEfficiency

        outputs['Mram'] = self.Mram
        outputs['Mmw'] = self.Mmw
        outputs['Pmw'] = self.Pmw

@dataclass
class Magnetorquers(om.ExplicitComponent):
    """Sizing of magnetic torquer"""

    T_total_mag: float
    B_mag: float
    I_mag: float
    A_mag: float
    r_mag: float
    rho_mag: float
    d_mag: float
    rho_wire: float

    def __init__(self, T_total_mag, B_mag, I_mag, A_mag, r_mag, rho_mag, d_mag, rho_wire):
        super().__init__()

        self.T_total_mag = T_total_mag
        self.B_mag = B_mag
        self.I_mag = I_mag
        self.A_mag = A_mag
        self.r_mag = r_mag
        self.rho_mag = rho_mag
        self.d_mag = d_mag
        self.rho_wire = rho_wire

    def setup(self):
        self.add_input('T_total_mag', val=self.T_total_mag, units="N*m")
        self.add_input('B_mag', val=self.B_mag, units="T")
        self.add_input('I_mag', val=self.I_mag, units="A")
        self.add_input('A_mag', val=self.A_mag, units="m**2")
        self.add_input('r_mag', val=self.r_mag, units="m")
        self.add_input('rho_mag', val=self.rho_mag)
        self.add_input('d_mag', val=self.d_mag, units="m")
        self.add_input('rho_wire', val=self.rho_wire, units="kg/m**3")

        self.add_output('D_mag', units="A*m**2")
        self.add_output('N_mag')
        self.add_output('L_mag', units="m")
        self.add_output('R_mag')
        self.add_output('P_mag', units="W")
        self.add_output('m_mag', units="kg")

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):
        self.T_total_mag = inputs['T_total_mag']
        self.B_mag = inputs['B_mag']
        self.I_mag = inputs['I_mag']
        self.A_mag = inputs['A_mag']
        self.r_mag = inputs['r_mag']
        self.rho_mag = inputs['rho_mag']
        self.d_mag = inputs['d_mag']
        self.rho_wire = inputs['rho_wire']

        self.D_mag = self.T_total_mag / self.B_mag
        self.N_mag = self.D_mag / (self.I_mag * self.A_mag)
        self.L_mag = 2 * math.pi * self.r_mag * self.N_mag
        Ac = math.pi * (self.d_mag / 2) ** 2
        self.R_mag = self.rho_mag * self.L_mag / Ac
        self.P_mag = self.I_mag ** 2 * self.R_mag
        self.m_mag = self.rho_wire * self.L_mag * Ac

        outputs['D_mag'] = self.D_mag
        outputs['N_mag'] = self.N_mag
        outputs['L_mag'] = self.L_mag
        outputs['R_mag'] = self.R_mag
        outputs['P_mag'] = self.P_mag
        outputs['m_mag'] = self.m_mag


"""
4. Power Subsystem: Solar Panel, Secondary Battery, and Primary Battery
"""

@dataclass
class SolarPanel(om.ExplicitComponent):
    """Solar panel sizing"""

    f_sunlight: float
    PDPY: float
    SF: float
    Efficiency_sp: float
    Wcsia: float
    Area_density: float

    def __init__(self, f_sunlight, PDPY, SF, Efficiency_sp, Wcsia, Area_density):
        super().__init__()

        self.f_sunlight = f_sunlight
        self.PDPY = PDPY
        self.SF = SF
        self.Efficiency_sp = Efficiency_sp
        self.Wcsia = Wcsia
        self.Area_density = Area_density

    def setup(self):
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('f_sunlight', val=self.f_sunlight)
        self.add_input('PDPY', val=self.PDPY)
        self.add_input('SF', val=self.SF, units="W/m**2")
        self.add_input('Efficiency_sp', val=self.Efficiency_sp)
        self.add_input('Wcsia', val=self.Wcsia, units="deg")
        self.add_input('Area_density', val=self.Area_density, units="kg/m**2")

        self.add_output('P_generated', units="W")
        self.add_output('A_solarpanel', units="m**2")
        self.add_output('m_solarpanel', units="kg")  # shared variable

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        P_total = inputs['P_total']
        self.f_sunlight = inputs['f_sunlight']
        self.PDPY = inputs['PDPY']
        self.SF = inputs['SF']
        self.Efficiency_sp = inputs['Efficiency_sp']
        self.Wcsia = inputs['Wcsia']
        self.Area_density = inputs['Area_density']

        P_average = P_total / self.f_sunlight
        self.P_generated = P_average / (1 - self.PDPY)
        Wcsia_deg = np.radians(self.Wcsia)
        P_area = self.SF * self.Efficiency_sp * np.cos(Wcsia_deg)
        self.A_solarpanel = self.P_generated / P_area
        self.m_solarpanel = self.Area_density * self.A_solarpanel

        outputs['P_generated'] = self.P_generated
        outputs['A_solarpanel'] = self.A_solarpanel
        outputs['m_solarpanel'] = self.m_solarpanel

@dataclass
class SolarPanelGaAs(om.ExplicitComponent):
    """Solar panel sizing"""

    f_sunlight_GaAs: float
    PDPY_GaAs: float
    SF_GaAs: float
    Efficiency_sp_GaAs: float
    Wcsia_GaAs: float
    Area_density_GaAs: float

    def __init__(
            self,
            f_sunlight_GaAs,
            PDPY_GaAs,
            SF_GaAs,
            Efficiency_sp_GaAs,
            Wcsia_GaAs,
            Area_density_GaAs
    ):
        super().__init__()

        # Store constructor arguments
        self.f_sunlight_GaAs = f_sunlight_GaAs
        self.PDPY_GaAs = PDPY_GaAs
        self.SF_GaAs = SF_GaAs
        self.Efficiency_sp_GaAs = Efficiency_sp_GaAs
        self.Wcsia_GaAs = Wcsia_GaAs
        self.Area_density_GaAs = Area_density_GaAs

    def setup(self):
        # Inputs
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('f_sunlight_GaAs', val=self.f_sunlight_GaAs)
        self.add_input('PDPY_GaAs', val=self.PDPY_GaAs)
        self.add_input('SF_GaAs', val=self.SF_GaAs, units="W/m**2")
        self.add_input('Efficiency_sp_GaAs', val=self.Efficiency_sp_GaAs)
        self.add_input('Wcsia_GaAs', val=self.Wcsia_GaAs, units="deg")
        self.add_input('Area_density_GaAs', val=self.Area_density_GaAs, units="kg/m**2")

        # Outputs
        self.add_output('P_generated', units="W")
        self.add_output('A_solarpanel', units="m**2")
        self.add_output('m_solarpanel', units="kg")  # shared variable

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        P_total_GaAs = inputs['P_total']
        self.f_sunlight_GaAs = inputs['f_sunlight_GaAs']
        self.PDPY_GaAs = inputs['PDPY_GaAs']
        self.SF_GaAs = inputs['SF_GaAs']
        self.Efficiency_sp_GaAs = inputs['Efficiency_sp_GaAs']
        self.Wcsia_GaAs = inputs['Wcsia_GaAs']
        self.Area_density_GaAs = inputs['Area_density_GaAs']

        # Perform calculations
        P_average = P_total_GaAs / self.f_sunlight_GaAs
        P_generated = P_average / (1 - self.PDPY_GaAs)
        Wcsia_rad = np.radians(self.Wcsia_GaAs)
        P_area = self.SF_GaAs * self.Efficiency_sp_GaAs * np.cos(Wcsia_rad)
        A_solarpanel = P_generated / P_area
        m_solarpanel = self.Area_density_GaAs * A_solarpanel

        outputs['P_generated'] = P_generated
        outputs['A_solarpanel'] = A_solarpanel
        outputs['m_solarpanel'] = m_solarpanel

@dataclass
class SolarPanelSi(om.ExplicitComponent):
    """Solar panel sizing"""

    f_sunlight_Si: float
    PDPY_Si: float
    SF_Si: float
    Efficiency_sp_Si: float
    Wcsia_Si: float
    Area_density_Si: float

    def __init__(
            self,
            f_sunlight_Si,
            PDPY_Si,
            SF_Si,
            Efficiency_sp_Si,
            Wcsia_Si,
            Area_density_Si
    ):
        super().__init__()

        # Store constructor arguments
        self.f_sunlight_Si = f_sunlight_Si
        self.PDPY_Si = PDPY_Si
        self.SF_Si = SF_Si
        self.Efficiency_sp_Si = Efficiency_sp_Si
        self.Wcsia_Si = Wcsia_Si
        self.Area_density_Si = Area_density_Si

    def setup(self):
        # Inputs
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('f_sunlight_Si', val=self.f_sunlight_Si)
        self.add_input('PDPY_Si', val=self.PDPY_Si)
        self.add_input('SF_Si', val=self.SF_Si, units="W/m**2")
        self.add_input('Efficiency_sp_Si', val=self.Efficiency_sp_Si)
        self.add_input('Wcsia_Si', val=self.Wcsia_Si, units="deg")
        self.add_input('Area_density_Si', val=self.Area_density_Si, units="kg/m**2")

        # Outputs
        self.add_output('P_generated', units="W")
        self.add_output('A_solarpanel', units="m**2")
        self.add_output('m_solarpanel', units="kg")  # shared variable

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        P_total_Si = inputs['P_total']
        self.f_sunlight_Si = inputs['f_sunlight_Si']
        self.PDPY_Si = inputs['PDPY_Si']
        self.SF_Si = inputs['SF_Si']
        self.Efficiency_sp_Si = inputs['Efficiency_sp_Si']
        self.Wcsia_Si = inputs['Wcsia_Si']
        self.Area_density_Si = inputs['Area_density_Si']

        # Perform calculations
        P_average = P_total_Si / self.f_sunlight_Si
        P_generated = P_average / (1 - self.PDPY_Si)
        Wcsia_rad = np.radians(self.Wcsia_Si)
        P_area = self.SF_Si * self.Efficiency_sp_Si * np.cos(Wcsia_rad)
        A_solarpanel = P_generated / P_area
        m_solarpanel = self.Area_density_Si * A_solarpanel

        outputs['P_generated'] = P_generated
        outputs['A_solarpanel'] = A_solarpanel
        outputs['m_solarpanel'] = m_solarpanel

@dataclass
class SolarPanelMJ(om.ExplicitComponent):
    """Solar panel sizing"""

    f_sunlight_MJ: float
    PDPY_MJ: float
    SF_MJ: float
    Efficiency_sp_MJ: float
    Wcsia_MJ: float
    Area_density_MJ: float

    def __init__(
            self,
            f_sunlight_MJ,
            PDPY_MJ,
            SF_MJ,
            Efficiency_sp_MJ,
            Wcsia_MJ,
            Area_density_MJ
    ):
        super().__init__()

        # Store constructor arguments
        self.f_sunlight_MJ = f_sunlight_MJ
        self.PDPY_MJ = PDPY_MJ
        self.SF_MJ = SF_MJ
        self.Efficiency_sp_MJ = Efficiency_sp_MJ
        self.Wcsia_MJ = Wcsia_MJ
        self.Area_density_MJ = Area_density_MJ

    def setup(self):
        # Inputs
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('f_sunlight_MJ', val=self.f_sunlight_MJ)
        self.add_input('PDPY_MJ', val=self.PDPY_MJ)
        self.add_input('SF_MJ', val=self.SF_MJ, units="W/m**2")
        self.add_input('Efficiency_sp_MJ', val=self.Efficiency_sp_MJ)
        self.add_input('Wcsia_MJ', val=self.Wcsia_MJ, units="deg")
        self.add_input('Area_density_MJ', val=self.Area_density_MJ, units="kg/m**2")

        # Outputs
        self.add_output('P_generated', units="W")
        self.add_output('A_solarpanel', units="m**2")
        self.add_output('m_solarpanel', units="kg")  # shared variable

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        P_total_MJ = inputs['P_total']
        self.f_sunlight_MJ = inputs['f_sunlight_MJ']
        self.PDPY_MJ = inputs['PDPY_MJ']
        self.SF_MJ = inputs['SF_MJ']
        self.Efficiency_sp_MJ = inputs['Efficiency_sp_MJ']
        self.Wcsia_MJ = inputs['Wcsia_MJ']
        self.Area_density_MJ = inputs['Area_density_MJ']

        # Perform calculations
        P_average = P_total_MJ / self.f_sunlight_MJ
        P_generated = P_average / (1 - self.PDPY_MJ)
        Wcsia_rad = np.radians(self.Wcsia_MJ)
        P_area = self.SF_MJ * self.Efficiency_sp_MJ * np.cos(Wcsia_rad)
        A_solarpanel = P_generated / P_area
        m_solarpanel = self.Area_density_MJ * A_solarpanel

        outputs['P_generated'] = P_generated
        outputs['A_solarpanel'] = A_solarpanel
        outputs['m_solarpanel'] = m_solarpanel

@dataclass
class SecondaryBattery(om.ExplicitComponent):
    """Secondary battery sizing"""

    MaxET: float
    DoD: float
    Ed: float

    # E_battery_sb: float
    # Msb: float

    def __init__(self, MaxET, DoD, Ed):
        super().__init__()

        self.MaxET = MaxET
        self.DoD = DoD
        self.Ed = Ed
        # self.E_battery_sb = E_battery_sb
        # self.Msb = Msb

    def setup(self):
        self.add_input('MaxET', val=self.MaxET, units="min")
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('DoD', val=self.DoD)
        self.add_input('Ed', val=self.Ed)

        self.add_output('E_battery_sb')
        self.add_output('Msb', units="kg")  # shared variable

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.MaxET = inputs['MaxET']
        P_total = inputs['P_total']
        self.DoD = inputs['DoD']
        self.Ed = inputs['Ed']

        Margin = 0.5
        E_eclipse = (1 + Margin) * P_total * self.MaxET * 60 / 3600
        self.E_battery_sb = E_eclipse / self.DoD
        self.Msb = self.E_battery_sb / self.Ed

        outputs['E_battery_sb'] = self.E_battery_sb
        outputs['Msb'] = self.Msb

@dataclass
class SecondaryBatteryLiIon(om.ExplicitComponent):
    """Secondary battery sizing"""

    MaxET_LiIon: float
    DoD_LiIon: float
    Ed_LiIon: float

    def __init__(self, MaxET_LiIon, DoD_LiIon, Ed_LiIon):
        super().__init__()

        self.MaxET_LiIon = MaxET_LiIon
        self.DoD_LiIon = DoD_LiIon
        self.Ed_LiIon = Ed_LiIon

    def setup(self):
        # Inputs (adding _LiIon suffix)
        self.add_input('MaxET_LiIon', val=self.MaxET_LiIon, units="min")
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('DoD_LiIon', val=self.DoD_LiIon)
        self.add_input('Ed_LiIon', val=self.Ed_LiIon)

        # Outputs (unchanged)
        self.add_output('E_battery_sb')
        self.add_output('Msb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MaxET_LiIon = inputs['MaxET_LiIon']
        P_total = inputs['P_total']
        DoD_LiIon = inputs['DoD_LiIon']
        Ed_LiIon = inputs['Ed_LiIon']

        # Perform calculations
        Margin = 0.5
        E_eclipse = (1 + Margin) * P_total * MaxET_LiIon * 60 / 3600
        E_battery_sb = E_eclipse / DoD_LiIon
        Msb = E_battery_sb / Ed_LiIon

        # Assign outputs
        outputs['E_battery_sb'] = E_battery_sb
        outputs['Msb'] = Msb

@dataclass
class SecondaryBatteryNiCd(om.ExplicitComponent):
    """Secondary battery sizing"""

    MaxET_NiCd: float
    DoD_NiCd: float
    Ed_NiCd: float

    def __init__(self, MaxET_NiCd, DoD_NiCd, Ed_NiCd):
        super().__init__()

        self.MaxET_NiCd = MaxET_NiCd
        self.DoD_NiCd = DoD_NiCd
        self.Ed_NiCd = Ed_NiCd

    def setup(self):
        # Inputs (adding _NiCd suffix)
        self.add_input('MaxET_NiCd', val=self.MaxET_NiCd, units="min")
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('DoD_NiCd', val=self.DoD_NiCd)
        self.add_input('Ed_NiCd', val=self.Ed_NiCd)

        # Outputs (unchanged)
        self.add_output('E_battery_sb')
        self.add_output('Msb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MaxET_NiCd = inputs['MaxET_NiCd']
        P_total = inputs['P_total']
        DoD_NiCd = inputs['DoD_NiCd']
        Ed_NiCd = inputs['Ed_NiCd']

        # Perform calculations
        Margin = 0.5
        E_eclipse = (1 + Margin) * P_total * MaxET_NiCd * 60 / 3600
        E_battery_sb = E_eclipse / DoD_NiCd
        Msb = E_battery_sb / Ed_NiCd

        # Assign outputs
        outputs['E_battery_sb'] = E_battery_sb
        outputs['Msb'] = Msb

@dataclass
class SecondaryBatteryNa2S(om.ExplicitComponent):
    """Secondary battery sizing"""

    MaxET_Na2S: float
    DoD_Na2S: float
    Ed_Na2S: float

    def __init__(self, MaxET_Na2S, DoD_Na2S, Ed_Na2S):
        super().__init__()

        self.MaxET_Na2S = MaxET_Na2S
        self.DoD_Na2S = DoD_Na2S
        self.Ed_Na2S = Ed_Na2S

    def setup(self):
        # Inputs (adding _Na2S suffix)
        self.add_input('MaxET_Na2S', val=self.MaxET_Na2S, units="min")
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('DoD_Na2S', val=self.DoD_Na2S)
        self.add_input('Ed_Na2S', val=self.Ed_Na2S)

        # Outputs (unchanged)
        self.add_output('E_battery_sb')
        self.add_output('Msb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MaxET_Na2S = inputs['MaxET_Na2S']
        P_total = inputs['P_total']
        DoD_Na2S = inputs['DoD_Na2S']
        Ed_Na2S = inputs['Ed_Na2S']

        # Perform calculations
        Margin = 0.5
        E_eclipse = (1 + Margin) * P_total * MaxET_Na2S * 60 / 3600
        E_battery_sb = E_eclipse / DoD_Na2S
        Msb = E_battery_sb / Ed_Na2S

        # Assign outputs
        outputs['E_battery_sb'] = E_battery_sb
        outputs['Msb'] = Msb

@dataclass
class PrimaryBattery(om.ExplicitComponent):
    """Primary battery sizing"""

    MD: float
    e_pb: float

    def __init__(self, MD, e_pb):
        super().__init__()

        self.MD = MD
        self.e_pb = e_pb

    def setup(self):
        self.add_input('MD', val=self.MD)
        self.add_input('P_total', units="W")  # shared variable
        self.add_input('e_pb', val=self.e_pb)

        self.add_output('E_battery_pb')
        self.add_output('Mpb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.MD = inputs['MD']
        P_total = inputs['P_total']
        self.e_pb = inputs['e_pb']

        Margin = 0.2
        self.E_battery_pb = (1 + Margin) * P_total * self.MD * 24
        self.Mpb = self.E_battery_pb / self.e_pb

        outputs['E_battery_pb'] = self.E_battery_pb
        outputs['Mpb'] = self.Mpb

@dataclass
class PrimaryBatteryAgZn(om.ExplicitComponent):
    """Primary battery sizing"""

    MD_AgZn: float
    e_pb_AgZn: float

    def __init__(self, MD_AgZn, e_pb_AgZn):
        super().__init__()

        self.MD_AgZn = MD_AgZn
        self.e_pb_AgZn = e_pb_AgZn

    def setup(self):
        # Inputs (with _AgZn for MD and e_pb; P_total remains unchanged)
        self.add_input('MD_AgZn', val=self.MD_AgZn)
        self.add_input('P_total', units="W")  # shared variable (unchanged)
        self.add_input('e_pb_AgZn', val=self.e_pb_AgZn)

        # Outputs (unchanged)
        self.add_output('E_battery_pb')
        self.add_output('Mpb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MD_AgZn = inputs['MD_AgZn']
        P_total = inputs['P_total']  # unchanged
        e_pb_AgZn = inputs['e_pb_AgZn']

        # Primary battery calculation
        Margin = 0.2
        E_battery_pb = (1 + Margin) * P_total * MD_AgZn * 24
        Mpb = E_battery_pb / e_pb_AgZn

        # Assign outputs
        outputs['E_battery_pb'] = E_battery_pb
        outputs['Mpb'] = Mpb

@dataclass
class PrimaryBatteryLiSOCl2(om.ExplicitComponent):
    """Primary battery sizing"""

    MD_LiSOCl2: float
    e_pb_LiSOCl2: float

    def __init__(self, MD_LiSOCl2, e_pb_LiSOCl2):
        super().__init__()

        self.MD_LiSOCl2 = MD_LiSOCl2
        self.e_pb_LiSOCl2 = e_pb_LiSOCl2

    def setup(self):
        # Inputs (with _LiSOCl2 for MD and e_pb; P_total remains unchanged)
        self.add_input('MD_LiSOCl2', val=self.MD_LiSOCl2)
        self.add_input('P_total', units="W")  # shared variable (unchanged)
        self.add_input('e_pb_LiSOCl2', val=self.e_pb_LiSOCl2)

        # Outputs (unchanged)
        self.add_output('E_battery_pb')
        self.add_output('Mpb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MD_LiSOCl2 = inputs['MD_LiSOCl2']
        P_total = inputs['P_total']  # unchanged
        e_pb_LiSOCl2 = inputs['e_pb_LiSOCl2']

        # Primary battery calculation
        Margin = 0.2
        E_battery_pb = (1 + Margin) * P_total * MD_LiSOCl2 * 24
        Mpb = E_battery_pb / e_pb_LiSOCl2

        # Assign outputs
        outputs['E_battery_pb'] = E_battery_pb
        outputs['Mpb'] = Mpb

@dataclass
class PrimaryBatteryLiCl(om.ExplicitComponent):
    """Primary battery sizing"""

    MD_LiCl: float
    e_pb_LiCl: float

    def __init__(self, MD_LiCl, e_pb_LiCl):
        super().__init__()

        self.MD_LiCl = MD_LiCl
        self.e_pb_LiCl = e_pb_LiCl

    def setup(self):
        # Inputs (with _LiCl for MD and e_pb; P_total remains unchanged)
        self.add_input('MD_LiCl', val=self.MD_LiCl)
        self.add_input('P_total', units="W")  # shared variable (unchanged)
        self.add_input('e_pb_LiCl', val=self.e_pb_LiCl)

        # Outputs (unchanged)
        self.add_output('E_battery_pb')
        self.add_output('Mpb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        # Retrieve inputs
        MD_LiCl = inputs['MD_LiCl']
        P_total = inputs['P_total']  # unchanged
        e_pb_LiCl = inputs['e_pb_LiCl']

        # Primary battery calculation
        Margin = 0.2
        E_battery_pb = (1 + Margin) * P_total * MD_LiCl * 24
        Mpb = E_battery_pb / e_pb_LiCl

        # Assign outputs
        outputs['E_battery_pb'] = E_battery_pb
        outputs['Mpb'] = Mpb


"""
5. Payload Subsystem: Remote Sensing
"""

@dataclass
class PayloadRemoteSensing(om.ExplicitComponent):
    """Sizing of payload for remote sensing"""

    x_optical: float
    Alt_payload: float
    GSD_optical: float
    lambda_optical: float
    Q_optical: float

    def __init__(self, x_optical, Alt_payload, GSD_optical, lambda_optical, Q_optical):
        super().__init__()

        self.x_optical = x_optical
        self.Alt_payload = Alt_payload
        self.GSD_optical = GSD_optical
        self.lambda_optical = lambda_optical
        self.Q_optical = Q_optical

    def setup(self):
        self.add_input('x_optical', val=self.x_optical, units="m", desc="Pixel size")
        self.add_input('Alt_payload', val=self.Alt_payload, units="km", desc="Orbit altitude")
        self.add_input('GSD_optical', val=self.GSD_optical, units="m", desc="Ground Sampling Distance")
        self.add_input('lambda_optical', val=self.lambda_optical, units="m", desc="Wave length")
        self.add_input('Q_optical', val=self.Q_optical, desc="Image quality")

        self.add_output('f_optical', units="mm", desc="Focal length")
        self.add_output('D_optical', units="mm", desc="Aperture diameter")
        self.add_output('m_payload', units="kg")  # shared variable
        self.add_output('P_payload', units="W")  # shared variable
        self.add_output('l_payload', units="m", desc="length of payload")
        self.add_output('w_payload', units="m", desc="width of payload")
        self.add_output('h_payload', units="m", desc="height of payload")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.x_optical = inputs['x_optical']
        self.Alt_payload = inputs['Alt_payload']
        self.GSD_optical = inputs['GSD_optical']
        self.lambda_optical = inputs['lambda_optical']
        self.Q_optical = inputs['Q_optical']

        self.f_optical = self.x_optical * self.Alt_payload * 1000 / self.GSD_optical
        self.D_optical = (self.lambda_optical * self.f_optical) / (self.Q_optical * self.x_optical)

        R = self.f_optical * 1000 / 70
        K = 1
        self.m_payload = K * R ** 3 * 0.277
        self.P_payload = K * R ** 3 * 1.3

        self.l_payload = R * 0.096
        self.w_payload = R * 0.090
        self.h_payload = R * 0.058

        outputs['f_optical'] = self.f_optical
        outputs['D_optical'] = self.D_optical
        outputs['m_payload'] = self.m_payload
        outputs['P_payload'] = self.P_payload
        outputs['l_payload'] = self.l_payload
        outputs['w_payload'] = self.w_payload
        outputs['h_payload'] = self.h_payload

@dataclass
class PayloadRemoteSensingNew(om.ExplicitComponent):
    """Sizing of payload for remote sensing"""

    x_optical: float
    Alt_payload: float
    #GSD_optical: float
    lambda_optical: float
    Q_optical: float

    def __init__(self, x_optical, Alt_payload, lambda_optical, Q_optical):
        super().__init__()

        self.x_optical = x_optical
        self.Alt_payload = Alt_payload
        #self.GSD_optical = GSD_optical
        self.lambda_optical = lambda_optical
        self.Q_optical = Q_optical

    def setup(self):
        self.add_input('x_optical', val=self.x_optical, units="m", desc="Pixel size")
        self.add_input('Alt_payload', val=self.Alt_payload, units="km", desc="Orbit altitude")
        self.add_input('GSD_optical', val=30.0, units="m", desc="Ground Sampling Distance")
        self.add_input('lambda_optical', val=self.lambda_optical, units="m", desc="Wave length")
        self.add_input('Q_optical', val=self.Q_optical, desc="Image quality")

        self.add_output('f_optical', units="mm", desc="Focal length")
        self.add_output('D_optical', units="mm", desc="Aperture diameter")
        self.add_output('m_payload', units="kg")  # shared variable
        self.add_output('P_payload', units="W")  # shared variable
        self.add_output('l_payload', units="m", desc="length of payload")
        self.add_output('w_payload', units="m", desc="width of payload")
        self.add_output('h_payload', units="m", desc="height of payload")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.x_optical = inputs['x_optical']
        self.Alt_payload = inputs['Alt_payload']
        GSD_optical = inputs['GSD_optical']
        self.lambda_optical = inputs['lambda_optical']
        self.Q_optical = inputs['Q_optical']

        self.f_optical = self.x_optical * self.Alt_payload * 1000 / GSD_optical
        self.D_optical = (self.lambda_optical * self.f_optical) / (self.Q_optical * self.x_optical)

        R = self.x_optical * self.Alt_payload * 1000 / GSD_optical * 1000 / 70
        K = 1
        self.m_payload = K * R ** 3 * 0.277
        self.P_payload = K * R ** 3 * 1.3

        self.l_payload = R * 0.096
        self.w_payload = R * 0.090
        self.h_payload = R * 0.058

        outputs['f_optical'] = self.f_optical
        outputs['D_optical'] = self.D_optical
        outputs['m_payload'] = self.m_payload
        outputs['P_payload'] = self.P_payload
        outputs['l_payload'] = self.l_payload
        outputs['w_payload'] = self.w_payload
        outputs['h_payload'] = self.h_payload


"""
6. Communication Subsystem: Radio Transceiver 
"""

@dataclass
class CommunicationRadioTransceiver(om.ExplicitComponent):
    """Sizing model of radio transceiver as communication component"""

    f_down: float
    P_t_down: float
    L_l_down: float
    theta_t_down: float
    e_t_down: float
    S_down: float
    L_a_down: float
    eff_down: float
    D_r_down: float
    e_r_down: float
    T_s_down: float
    R_down: float
    BER_down: float
    L_imp_down: float
    Eb_No_req_down: float

    # M_comm_down: float
    # P_comm_down: float
    # Br_down: float

    def __init__(self, f_down, P_t_down, L_l_down, theta_t_down, e_t_down, S_down, L_a_down,
                 eff_down, D_r_down, e_r_down, T_s_down, R_down, BER_down, L_imp_down, Eb_No_req_down):
        super().__init__()

        self.f_down = f_down
        self.P_t_down = P_t_down
        self.L_l_down = L_l_down
        self.theta_t_down = theta_t_down
        self.e_t_down = e_t_down
        self.S_down = S_down
        self.L_a_down = L_a_down
        self.eff_down = eff_down
        self.D_r_down = D_r_down
        self.e_r_down = e_r_down
        self.T_s_down = T_s_down
        self.R_down = R_down
        self.BER_down = BER_down
        self.L_imp_down = L_imp_down
        self.Eb_No_req_down = Eb_No_req_down
        # self.M_comm_down = M_comm_down
        # self.P_comm_down = P_comm_down
        # self.Br_down = Br_down

    def setup(self):
        self.add_input('f_down', val=self.f_down, units="Hz", desc="Carrier frequency")
        self.add_input('P_t_down', val=self.P_t_down, units="W", desc="Transmitter power")
        self.add_input('L_l_down', val=self.L_l_down, desc="Transmitter line loss")
        self.add_input('theta_t_down', val=self.theta_t_down, units="deg", desc="Transmit antenna beamwidth")
        self.add_input('e_t_down', val=self.e_t_down, units="deg", desc="Transmit antenna pointing offset")
        self.add_input('S_down', val=self.S_down, units="km", desc="Propagation path length")
        self.add_input('L_a_down', val=self.L_a_down, desc="Propagation and polarization loss")  # might be an output
        self.add_input('eff_down', val=self.eff_down, desc="Antenna efficiency")
        self.add_input('D_r_down', val=self.D_r_down, units="m", desc="Receive antenna diameter")
        self.add_input('e_r_down', val=self.e_r_down, units="deg", desc="Receive antenna pointing error")
        self.add_input('T_s_down', val=self.T_s_down, units="K", desc="System noise temperature")  # might be an output
        self.add_input('R_down', val=self.R_down, units="s**(-1)", desc="Data rate")
        self.add_input('BER_down', val=self.BER_down, desc="Bit Error Rate")
        self.add_input('L_imp_down', val=self.L_imp_down, desc="Implementation loss")
        self.add_input('Eb_No_req_down', val=self.Eb_No_req_down,
                       desc="Required system-to-noise ratio")  # might be an output

        # Outputs
        self.add_output('M_comm_down', units="kg", desc="Mass communication subsystem")  # shared var
        self.add_output('P_comm_down', units="W", desc="DC input Power communication subsystem")  # shared var

        # New outputs
        self.add_output('G_pt_down', desc="Peak transmit antenna gain")
        self.add_output('D_t_down', units="m", desc="Transmit antenna diameter")
        self.add_output('L_pt_down', desc="Transmit antenna pointing loss")
        self.add_output('G_t_down', desc="Transmit antenna gain (net)")
        self.add_output('EIRP_down', desc="Equivalence Isotropic Radiated Power")
        self.add_output('L_s_down', desc="Space loss")
        self.add_output('G_rp_down', desc="Peak receive antenna gain (net)")
        self.add_output('theta_r_down', units="deg", desc="Receive antenna beamwidth")
        self.add_output('L_pr_down', desc="Receive antenna pointing loss")
        self.add_output('G_r_down', desc="Receive antenna gain")
        self.add_output('Eb/No_down', desc="System-to-noise ratio")
        self.add_output('C/No_down', desc="Carrier-to-noise density ratio")
        self.add_output('Margin_down', desc="Margin")
        self.add_output('data_downloaded', desc="data downloaded")

        # Derivatives declaration
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.f_down = inputs['f_down']
        self.P_t_down = inputs['P_t_down']
        self.L_l_down = inputs['L_l_down']
        self.theta_t_down = inputs['theta_t_down']
        self.e_t_down = inputs['e_t_down']
        self.S_down = inputs['S_down']
        self.L_a_down = inputs['L_a_down']
        self.eff_down = inputs['eff_down']
        self.D_r_down = inputs['D_r_down']
        self.e_r_down = inputs['e_r_down']
        self.T_s_down = inputs['T_s_down']
        self.R_down = inputs['R_down']
        self.BER_down = inputs['BER_down']
        self.L_imp_down = inputs['L_imp_down']
        self.Eb_No_req_down = inputs['Eb_No_req_down']

        # Sizing model
        P_t_dB = 10 * np.log10(self.P_t_down)  # transmitter power conversion to dB
        G_pt = 44.3 - 10 * np.log10(self.theta_t_down ** 2)  # peak transmit antenna gain (eq. 13-20)
        D_t = 21 / (self.f_down * self.theta_t_down)  # transmit antenna diameter (eq. 13-19)
        L_pt = -12 * (self.e_t_down / self.theta_t_down) ** 2  # transmit antenna pointing loss (eq. 13-21)
        G_t = G_pt + L_pt  # transmit antenna gain
        EIRP = P_t_dB + self.L_l_down + G_t  # equivalence isotropic radiated power
        L_s = 20 * np.log10(3e8) - 20 * np.log10(4 * np.pi) - 20 * np.log10(self.S_down * 1000) - 20 * np.log10(
            self.f_down) - 180.0  # Space loss
        G_rp = -159.59 + 20 * np.log10(self.D_r_down) + 20 * np.log10(self.f_down) + 10 * np.log10(
            self.eff_down) + 180.0  # peak receive antenna gain (eq. 13-18)
        theta_r = 21 / (self.f_down * self.D_r_down)  # receive antenna beamwidth (eq. 13-19)
        L_pr = -12 * (self.e_r_down / theta_r) ** 2  # receive antenna pointing loss (eq. 13-21)
        G_r = G_rp + L_pr  # receive antenna gain
        Eb_No = P_t_dB + self.L_l_down + G_t + L_pr + L_s + self.L_a_down + G_r + 228.6 - 10 * np.log10(
            self.T_s_down) - 10 * np.log10(self.R_down)
        C_No = Eb_No + 10 * np.log10(self.R_down)
        Margin = Eb_No - self.Eb_No_req_down + self.L_imp_down

        # Statistical approach, with the reference of EnduroSat p.259 SOTA of Small Spacecraft NASA
        R = self.f_down / 2.2
        self.M_comm_down = R ** 3 * 0.195
        self.P_comm_down = R ** 3 * 1.25

        self.data_downloaded = (3 * 10 ** 8 * G_r * self.L_l_down / (
                    16 * math.pi ** 2 * self.f_down * self.T_s_down * Eb_No)) * (
                                           self.eff_down * self.P_comm_down * G_r / self.S_down ** 2)

        outputs['M_comm_down'] = self.M_comm_down
        outputs['P_comm_down'] = self.P_comm_down
        outputs['data_downloaded'] = self.data_downloaded
        # New outputs
        outputs['G_pt_down'] = G_pt
        outputs['D_t_down'] = D_t
        outputs['L_pt_down'] = L_pt
        outputs['G_t_down'] = G_t
        outputs['EIRP_down'] = EIRP
        outputs['L_s_down'] = L_s
        outputs['G_rp_down'] = G_rp
        outputs['theta_r_down'] = theta_r
        outputs['L_pr_down'] = L_pr
        outputs['G_r_down'] = G_r
        outputs['Eb/No_down'] = Eb_No
        outputs['C/No_down'] = C_No
        outputs['Margin_down'] = Margin

@dataclass
class CommunicationRadioTransceiverUHF(om.ExplicitComponent):
    """Sizing model of radio transceiver as communication component"""

    f_down_UHF: float
    P_t_down_UHF: float
    L_l_down_UHF: float
    theta_t_down_UHF: float
    e_t_down_UHF: float
    S_down_UHF: float
    L_a_down_UHF: float
    eff_down_UHF: float
    D_r_down_UHF: float
    e_r_down_UHF: float
    T_s_down_UHF: float
    R_down_UHF: float
    BER_down_UHF: float
    L_imp_down_UHF: float
    Eb_No_req_down_UHF: float

    def __init__(self, f_down_UHF, P_t_down_UHF, L_l_down_UHF, theta_t_down_UHF, e_t_down_UHF, S_down_UHF, L_a_down_UHF,
                 eff_down_UHF, D_r_down_UHF, e_r_down_UHF, T_s_down_UHF, R_down_UHF, BER_down_UHF, L_imp_down_UHF,
                 Eb_No_req_down_UHF):
        super().__init__()

        self.f_down_UHF = f_down_UHF
        self.P_t_down_UHF = P_t_down_UHF
        self.L_l_down_UHF = L_l_down_UHF
        self.theta_t_down_UHF = theta_t_down_UHF
        self.e_t_down_UHF = e_t_down_UHF
        self.S_down_UHF = S_down_UHF
        self.L_a_down_UHF = L_a_down_UHF
        self.eff_down_UHF = eff_down_UHF
        self.D_r_down_UHF = D_r_down_UHF
        self.e_r_down_UHF = e_r_down_UHF
        self.T_s_down_UHF = T_s_down_UHF
        self.R_down_UHF = R_down_UHF
        self.BER_down_UHF = BER_down_UHF
        self.L_imp_down_UHF = L_imp_down_UHF
        self.Eb_No_req_down_UHF = Eb_No_req_down_UHF

    def setup(self):
        self.add_input('f_down_UHF', val=self.f_down_UHF, units="Hz", desc="Carrier frequency")
        self.add_input('P_t_down_UHF', val=self.P_t_down_UHF, units="W", desc="Transmitter power")
        self.add_input('L_l_down_UHF', val=self.L_l_down_UHF, desc="Transmitter line loss")
        self.add_input('theta_t_down_UHF', val=self.theta_t_down_UHF, units="deg", desc="Transmit antenna beamwidth")
        self.add_input('e_t_down_UHF', val=self.e_t_down_UHF, units="deg", desc="Transmit antenna pointing offset")
        self.add_input('S_down_UHF', val=self.S_down_UHF, units="km", desc="Propagation path length")
        self.add_input('L_a_down_UHF', val=self.L_a_down_UHF, desc="Propagation and polarization loss")
        self.add_input('eff_down_UHF', val=self.eff_down_UHF, desc="Antenna efficiency")
        self.add_input('D_r_down_UHF', val=self.D_r_down_UHF, units="m", desc="Receive antenna diameter")
        self.add_input('e_r_down_UHF', val=self.e_r_down_UHF, units="deg", desc="Receive antenna pointing error")
        self.add_input('T_s_down_UHF', val=self.T_s_down_UHF, units="K", desc="System noise temperature")
        self.add_input('R_down_UHF', val=self.R_down_UHF, units="s**(-1)", desc="Data rate")
        self.add_input('BER_down_UHF', val=self.BER_down_UHF, desc="Bit Error Rate")
        self.add_input('L_imp_down_UHF', val=self.L_imp_down_UHF, desc="Implementation loss")
        self.add_input('Eb_No_req_down_UHF', val=self.Eb_No_req_down_UHF, desc="Required system-to-noise ratio")

        self.add_output('M_comm_down', units="kg", desc="Mass communication subsystem")
        self.add_output('P_comm_down', units="W", desc="DC input Power communication subsystem")
        self.add_output('data_downloaded', desc="Data downloaded")

        self.add_output('G_pt_down', desc="Peak transmit antenna gain")
        self.add_output('D_t_down', units="m", desc="Transmit antenna diameter")
        self.add_output('L_pt_down', desc="Transmit antenna pointing loss")
        self.add_output('G_t_down', desc="Transmit antenna gain (net)")
        self.add_output('EIRP_down', desc="Equivalence Isotropic Radiated Power")
        self.add_output('L_s_down', desc="Space loss")
        self.add_output('G_rp_down', desc="Peak receive antenna gain (net)")
        self.add_output('theta_r_down', units="deg", desc="Receive antenna beamwidth")
        self.add_output('L_pr_down', desc="Receive antenna pointing loss")
        self.add_output('G_r_down', desc="Receive antenna gain")
        self.add_output('Eb/No_down', desc="System-to-noise ratio")
        self.add_output('C/No_down', desc="Carrier-to-noise density ratio")
        self.add_output('Margin_down', desc="Margin")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.f_down_UHF = inputs['f_down_UHF']
        self.P_t_down_UHF = inputs['P_t_down_UHF']
        self.L_l_down_UHF = inputs['L_l_down_UHF']
        self.theta_t_down_UHF = inputs['theta_t_down_UHF']
        self.e_t_down_UHF = inputs['e_t_down_UHF']
        self.S_down_UHF = inputs['S_down_UHF']
        self.L_a_down_UHF = inputs['L_a_down_UHF']
        self.eff_down_UHF = inputs['eff_down_UHF']
        self.D_r_down_UHF = inputs['D_r_down_UHF']
        self.e_r_down_UHF = inputs['e_r_down_UHF']
        self.T_s_down_UHF = inputs['T_s_down_UHF']
        self.R_down_UHF = inputs['R_down_UHF']
        self.BER_down_UHF = inputs['BER_down_UHF']
        self.L_imp_down_UHF = inputs['L_imp_down_UHF']
        self.Eb_No_req_down_UHF = inputs['Eb_No_req_down_UHF']

        # Sizing model
        P_t_dB = 10 * np.log10(self.P_t_down_UHF)  # transmitter power conversion to dB
        G_pt = 44.3 - 10 * np.log10(self.theta_t_down_UHF ** 2)  # peak transmit antenna gain (eq. 13-20)
        D_t = 21 / (self.f_down_UHF * self.theta_t_down_UHF)  # transmit antenna diameter (eq. 13-19)
        L_pt = -12 * (self.e_t_down_UHF / self.theta_t_down_UHF) ** 2  # transmit antenna pointing loss (eq. 13-21)
        G_t = G_pt + L_pt  # transmit antenna gain
        EIRP = P_t_dB + self.L_l_down_UHF + G_t  # equivalence isotropic radiated power
        L_s = 20 * np.log10(3e8) - 20 * np.log10(4 * np.pi) - 20 * np.log10(self.S_down_UHF * 1000) - 20 * np.log10(
            self.f_down_UHF) - 180.0  # Space loss
        G_rp = -159.59 + 20 * np.log10(self.D_r_down_UHF) + 20 * np.log10(self.f_down_UHF) + 10 * np.log10(
            self.eff_down_UHF) + 180.0  # peak receive antenna gain (eq. 13-18)
        theta_r = 21 / (self.f_down_UHF * self.D_r_down_UHF)  # receive antenna beamwidth (eq. 13-19)
        L_pr = -12 * (self.e_r_down_UHF / theta_r) ** 2  # receive antenna pointing loss (eq. 13-21)
        G_r = G_rp + L_pr  # receive antenna gain
        Eb_No = P_t_dB + self.L_l_down_UHF + G_t + L_pr + L_s + self.L_a_down_UHF + G_r + 228.6 - 10 * np.log10(
            self.T_s_down_UHF) - 10 * np.log10(self.R_down_UHF)
        C_No = Eb_No + 10 * np.log10(self.R_down_UHF)
        Margin = Eb_No - self.Eb_No_req_down_UHF + self.L_imp_down_UHF

        # Statistical approach, with the reference of EnduroSat p.259 SOTA of Small Spacecraft NASA
        R = self.f_down_UHF / 2.2
        self.M_comm_down = R ** 3 * 0.195
        self.P_comm_down = R ** 3 * 1.25

        self.data_downloaded = (3 * 10 ** 8 * G_r * self.L_l_down_UHF / (
                16 * math.pi ** 2 * self.f_down_UHF * self.T_s_down_UHF * Eb_No)) * (
                                       self.eff_down_UHF * self.P_comm_down * G_r / self.S_down_UHF ** 2)

        outputs['M_comm_down'] = self.M_comm_down
        outputs['P_comm_down'] = self.P_comm_down
        outputs['data_downloaded'] = self.data_downloaded
        # New outputs
        outputs['G_pt_down'] = G_pt
        outputs['D_t_down'] = D_t
        outputs['L_pt_down'] = L_pt
        outputs['G_t_down'] = G_t
        outputs['EIRP_down'] = EIRP
        outputs['L_s_down'] = L_s
        outputs['G_rp_down'] = G_rp
        outputs['theta_r_down'] = theta_r
        outputs['L_pr_down'] = L_pr
        outputs['G_r_down'] = G_r
        outputs['Eb/No_down'] = Eb_No
        outputs['C/No_down'] = C_No
        outputs['Margin_down'] = Margin

@dataclass
class CommunicationRadioTransceiverS(om.ExplicitComponent):
    """Sizing model of radio transceiver as communication component"""

    f_down_S: float
    P_t_down_S: float
    L_l_down_S: float
    theta_t_down_S: float
    e_t_down_S: float
    S_down_S: float
    L_a_down_S: float
    eff_down_S: float
    D_r_down_S: float
    e_r_down_S: float
    T_s_down_S: float
    R_down_S: float
    BER_down_S: float
    L_imp_down_S: float
    Eb_No_req_down_S: float

    def __init__(self, f_down_S, P_t_down_S, L_l_down_S, theta_t_down_S, e_t_down_S, S_down_S, L_a_down_S,
                 eff_down_S, D_r_down_S, e_r_down_S, T_s_down_S, R_down_S, BER_down_S, L_imp_down_S,
                 Eb_No_req_down_S):
        super().__init__()

        self.f_down_S = f_down_S
        self.P_t_down_S = P_t_down_S
        self.L_l_down_S = L_l_down_S
        self.theta_t_down_S = theta_t_down_S
        self.e_t_down_S = e_t_down_S
        self.S_down_S = S_down_S
        self.L_a_down_S = L_a_down_S
        self.eff_down_S = eff_down_S
        self.D_r_down_S = D_r_down_S
        self.e_r_down_S = e_r_down_S
        self.T_s_down_S = T_s_down_S
        self.R_down_S = R_down_S
        self.BER_down_S = BER_down_S
        self.L_imp_down_S = L_imp_down_S
        self.Eb_No_req_down_S = Eb_No_req_down_S

    def setup(self):
        self.add_input('f_down_S', val=self.f_down_S, units="Hz", desc="Carrier frequency")
        self.add_input('P_t_down_S', val=self.P_t_down_S, units="W", desc="Transmitter power")
        self.add_input('L_l_down_S', val=self.L_l_down_S, desc="Transmitter line loss")
        self.add_input('theta_t_down_S', val=self.theta_t_down_S, units="deg", desc="Transmit antenna beamwidth")
        self.add_input('e_t_down_S', val=self.e_t_down_S, units="deg", desc="Transmit antenna pointing offset")
        self.add_input('S_down_S', val=self.S_down_S, units="km", desc="Propagation path length")
        self.add_input('L_a_down_S', val=self.L_a_down_S, desc="Propagation and polarization loss")
        self.add_input('eff_down_S', val=self.eff_down_S, desc="Antenna efficiency")
        self.add_input('D_r_down_S', val=self.D_r_down_S, units="m", desc="Receive antenna diameter")
        self.add_input('e_r_down_S', val=self.e_r_down_S, units="deg", desc="Receive antenna pointing error")
        self.add_input('T_s_down_S', val=self.T_s_down_S, units="K", desc="System noise temperature")
        self.add_input('R_down_S', val=self.R_down_S, units="s**(-1)", desc="Data rate")
        self.add_input('BER_down_S', val=self.BER_down_S, desc="Bit Error Rate")
        self.add_input('L_imp_down_S', val=self.L_imp_down_S, desc="Implementation loss")
        self.add_input('Eb_No_req_down_S', val=self.Eb_No_req_down_S, desc="Required system-to-noise ratio")

        self.add_output('M_comm_down', units="kg", desc="Mass communication subsystem")
        self.add_output('P_comm_down', units="W", desc="DC input Power communication subsystem")
        self.add_output('data_downloaded', desc="Data downloaded")

        self.add_output('G_pt_down', desc="Peak transmit antenna gain")
        self.add_output('D_t_down', units="m", desc="Transmit antenna diameter")
        self.add_output('L_pt_down', desc="Transmit antenna pointing loss")
        self.add_output('G_t_down', desc="Transmit antenna gain (net)")
        self.add_output('EIRP_down', desc="Equivalence Isotropic Radiated Power")
        self.add_output('L_s_down', desc="Space loss")
        self.add_output('G_rp_down', desc="Peak receive antenna gain (net)")
        self.add_output('theta_r_down', units="deg", desc="Receive antenna beamwidth")
        self.add_output('L_pr_down', desc="Receive antenna pointing loss")
        self.add_output('G_r_down', desc="Receive antenna gain")
        self.add_output('Eb/No_down', desc="System-to-noise ratio")
        self.add_output('C/No_down', desc="Carrier-to-noise density ratio")
        self.add_output('Margin_down', desc="Margin")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.f_down_S = inputs['f_down_S']
        self.P_t_down_S = inputs['P_t_down_S']
        self.L_l_down_S = inputs['L_l_down_S']
        self.theta_t_down_S = inputs['theta_t_down_S']
        self.e_t_down_S = inputs['e_t_down_S']
        self.S_down_S = inputs['S_down_S']
        self.L_a_down_S = inputs['L_a_down_S']
        self.eff_down_S = inputs['eff_down_S']
        self.D_r_down_S = inputs['D_r_down_S']
        self.e_r_down_S = inputs['e_r_down_S']
        self.T_s_down_S = inputs['T_s_down_S']
        self.R_down_S = inputs['R_down_S']
        self.BER_down_S = inputs['BER_down_S']
        self.L_imp_down_S = inputs['L_imp_down_S']
        self.Eb_No_req_down_S = inputs['Eb_No_req_down_S']

        # Sizing model
        P_t_dB = 10 * np.log10(self.P_t_down_S)  # transmitter power conversion to dB
        G_pt = 44.3 - 10 * np.log10(self.theta_t_down_S ** 2)  # peak transmit antenna gain (eq. 13-20)
        D_t = 21 / (self.f_down_S * self.theta_t_down_S)  # transmit antenna diameter (eq. 13-19)
        L_pt = -12 * (self.e_t_down_S / self.theta_t_down_S) ** 2  # transmit antenna pointing loss (eq. 13-21)
        G_t = G_pt + L_pt  # transmit antenna gain
        EIRP = P_t_dB + self.L_l_down_S + G_t  # equivalence isotropic radiated power
        L_s = 20 * np.log10(3e8) - 20 * np.log10(4 * np.pi) - 20 * np.log10(self.S_down_S * 1000) - 20 * np.log10(
            self.f_down_S) - 180.0  # Space loss
        G_rp = -159.59 + 20 * np.log10(self.D_r_down_S) + 20 * np.log10(self.f_down_S) + 10 * np.log10(
            self.eff_down_S) + 180.0  # peak receive antenna gain (eq. 13-18)
        theta_r = 21 / (self.f_down_S * self.D_r_down_S)  # receive antenna beamwidth (eq. 13-19)
        L_pr = -12 * (self.e_r_down_S / theta_r) ** 2  # receive antenna pointing loss (eq. 13-21)
        G_r = G_rp + L_pr  # receive antenna gain
        Eb_No = P_t_dB + self.L_l_down_S + G_t + L_pr + L_s + self.L_a_down_S + G_r + 228.6 - 10 * np.log10(
            self.T_s_down_S) - 10 * np.log10(self.R_down_S)
        C_No = Eb_No + 10 * np.log10(self.R_down_S)
        Margin = Eb_No - self.Eb_No_req_down_S + self.L_imp_down_S

        # Statistical approach, with the reference of EnduroSat p.259 SOTA of Small Spacecraft NASA
        R = self.f_down_S / 2.2
        self.M_comm_down = R ** 3 * 0.195
        self.P_comm_down = R ** 3 * 1.25

        self.data_downloaded = (3 * 10 ** 8 * G_r * self.L_l_down_S / (
                    16 * math.pi ** 2 * self.f_down_S * self.T_s_down_S * Eb_No)) * (
                                           self.eff_down_S * self.P_comm_down * G_r / self.S_down_S ** 2)

        outputs['M_comm_down'] = self.M_comm_down
        outputs['P_comm_down'] = self.P_comm_down
        outputs['data_downloaded'] = self.data_downloaded
        # New outputs
        outputs['G_pt_down'] = G_pt
        outputs['D_t_down'] = D_t
        outputs['L_pt_down'] = L_pt
        outputs['G_t_down'] = G_t
        outputs['EIRP_down'] = EIRP
        outputs['L_s_down'] = L_s
        outputs['G_rp_down'] = G_rp
        outputs['theta_r_down'] = theta_r
        outputs['L_pr_down'] = L_pr
        outputs['G_r_down'] = G_r
        outputs['Eb/No_down'] = Eb_No
        outputs['C/No_down'] = C_No
        outputs['Margin_down'] = Margin

@dataclass
class CommunicationRadioTransceiverL(om.ExplicitComponent):
    """Sizing model of radio transceiver as communication component"""

    f_down_L: float
    P_t_down_L: float
    L_l_down_L: float
    theta_t_down_L: float
    e_t_down_L: float
    S_down_L: float
    L_a_down_L: float
    eff_down_L: float
    D_r_down_L: float
    e_r_down_L: float
    T_s_down_L: float
    R_down_L: float
    BER_down_L: float
    L_imp_down_L: float
    Eb_No_req_down_L: float

    def __init__(self, f_down_L, P_t_down_L, L_l_down_L, theta_t_down_L, e_t_down_L, S_down_L, L_a_down_L,
                 eff_down_L, D_r_down_L, e_r_down_L, T_s_down_L, R_down_L, BER_down_L, L_imp_down_L, Eb_No_req_down_L):
        super().__init__()

        self.f_down_L = f_down_L
        self.P_t_down_L = P_t_down_L
        self.L_l_down_L = L_l_down_L
        self.theta_t_down_L = theta_t_down_L
        self.e_t_down_L = e_t_down_L
        self.S_down_L = S_down_L
        self.L_a_down_L = L_a_down_L
        self.eff_down_L = eff_down_L
        self.D_r_down_L = D_r_down_L
        self.e_r_down_L = e_r_down_L
        self.T_s_down_L = T_s_down_L
        self.R_down_L = R_down_L
        self.BER_down_L = BER_down_L
        self.L_imp_down_L = L_imp_down_L
        self.Eb_No_req_down_L = Eb_No_req_down_L

    def setup(self):
        self.add_input('f_down_L', val=self.f_down_L, units="Hz", desc="Carrier frequency")
        self.add_input('P_t_down_L', val=self.P_t_down_L, units="W", desc="Transmitter power")
        self.add_input('L_l_down_L', val=self.L_l_down_L, desc="Transmitter line loss")
        self.add_input('theta_t_down_L', val=self.theta_t_down_L, units="deg", desc="Transmit antenna beamwidth")
        self.add_input('e_t_down_L', val=self.e_t_down_L, units="deg", desc="Transmit antenna pointing offset")
        self.add_input('S_down_L', val=self.S_down_L, units="km", desc="Propagation path length")
        self.add_input('L_a_down_L', val=self.L_a_down_L, desc="Propagation and polarization loss")
        self.add_input('eff_down_L', val=self.eff_down_L, desc="Antenna efficiency")
        self.add_input('D_r_down_L', val=self.D_r_down_L, units="m", desc="Receive antenna diameter")
        self.add_input('e_r_down_L', val=self.e_r_down_L, units="deg", desc="Receive antenna pointing error")
        self.add_input('T_s_down_L', val=self.T_s_down_L, units="K", desc="System noise temperature")
        self.add_input('R_down_L', val=self.R_down_L, units="s**(-1)", desc="Data rate")
        self.add_input('BER_down_L', val=self.BER_down_L, desc="Bit Error Rate")
        self.add_input('L_imp_down_L', val=self.L_imp_down_L, desc="Implementation loss")
        self.add_input('Eb_No_req_down_L', val=self.Eb_No_req_down_L, desc="Required system-to-noise ratio")

        # Outputs
        self.add_output('M_comm_down', units="kg", desc="Mass communication subsystem")  # shared var
        self.add_output('P_comm_down', units="W", desc="DC input Power communication subsystem")  # shared var

        # New outputs
        self.add_output('G_pt_down', desc="Peak transmit antenna gain")
        self.add_output('D_t_down', units="m", desc="Transmit antenna diameter")
        self.add_output('L_pt_down', desc="Transmit antenna pointing loss")
        self.add_output('G_t_down', desc="Transmit antenna gain (net)")
        self.add_output('EIRP_down', desc="Equivalence Isotropic Radiated Power")
        self.add_output('L_s_down', desc="Space loss")
        self.add_output('G_rp_down', desc="Peak receive antenna gain (net)")
        self.add_output('theta_r_down', units="deg", desc="Receive antenna beamwidth")
        self.add_output('L_pr_down', desc="Receive antenna pointing loss")
        self.add_output('G_r_down', desc="Receive antenna gain")
        self.add_output('Eb/No_down', desc="System-to-noise ratio")
        self.add_output('C/No_down', desc="Carrier-to-noise density ratio")
        self.add_output('Margin_down', desc="Margin")
        self.add_output('data_downloaded', desc="data downloaded")

        # Derivatives declaration
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.f_down_L = inputs['f_down_L']
        self.P_t_down_L = inputs['P_t_down_L']
        self.L_l_down_L = inputs['L_l_down_L']
        self.theta_t_down_L = inputs['theta_t_down_L']
        self.e_t_down_L = inputs['e_t_down_L']
        self.S_down_L = inputs['S_down_L']
        self.L_a_down_L = inputs['L_a_down_L']
        self.eff_down_L = inputs['eff_down_L']
        self.D_r_down_L = inputs['D_r_down_L']
        self.e_r_down_L = inputs['e_r_down_L']
        self.T_s_down_L = inputs['T_s_down_L']
        self.R_down_L = inputs['R_down_L']
        self.BER_down_L = inputs['BER_down_L']
        self.L_imp_down_L = inputs['L_imp_down_L']
        self.Eb_No_req_down_L = inputs['Eb_No_req_down_L']

        # Sizing model
        P_t_dB = 10 * np.log10(self.P_t_down_L)  # transmitter power conversion to dB
        G_pt = 44.3 - 10 * np.log10(self.theta_t_down_L ** 2)  # peak transmit antenna gain (eq. 13-20)
        D_t = 21 / (self.f_down_L * self.theta_t_down_L)  # transmit antenna diameter (eq. 13-19)
        L_pt = -12 * (self.e_t_down_L / self.theta_t_down_L) ** 2  # transmit antenna pointing loss (eq. 13-21)
        G_t = G_pt + L_pt  # transmit antenna gain
        EIRP = P_t_dB + self.L_l_down_L + G_t  # equivalence isotropic radiated power
        L_s = 20 * np.log10(3e8) - 20 * np.log10(4 * np.pi) - 20 * np.log10(self.S_down_L * 1000) - 20 * np.log10(
            self.f_down_L) - 180.0  # Space loss
        G_rp = -159.59 + 20 * np.log10(self.D_r_down_L) + 20 * np.log10(self.f_down_L) + 10 * np.log10(
            self.eff_down_L) + 180.0  # peak receive antenna gain (eq. 13-18)
        theta_r = 21 / (self.f_down_L * self.D_r_down_L)  # receive antenna beamwidth (eq. 13-19)
        L_pr = -12 * (self.e_r_down_L / theta_r) ** 2  # receive antenna pointing loss (eq. 13-21)
        G_r = G_rp + L_pr  # receive antenna gain
        Eb_No = P_t_dB + self.L_l_down_L + G_t + L_pr + L_s + self.L_a_down_L + G_r + 228.6 - 10 * np.log10(
            self.T_s_down_L) - 10 * np.log10(self.R_down_L)
        C_No = Eb_No + 10 * np.log10(self.R_down_L)
        Margin = Eb_No - self.Eb_No_req_down_L + self.L_imp_down_L

        # Statistical approach, with the reference of EnduroSat p.259 SOTA of Small Spacecraft NASA
        R = self.f_down_L / 2.2
        self.M_comm_down = R ** 3 * 0.195
        self.P_comm_down = R ** 3 * 1.25

        self.data_downloaded = (3 * 10 ** 8 * G_r * self.L_l_down_L / (
                16 * math.pi ** 2 * self.f_down_L * self.T_s_down_L * Eb_No)) * (
                                       self.eff_down_L * self.P_comm_down * G_r / self.S_down_L ** 2)

        outputs['M_comm_down'] = self.M_comm_down
        outputs['P_comm_down'] = self.P_comm_down
        outputs['data_downloaded'] = self.data_downloaded
        # New outputs
        outputs['G_pt_down'] = G_pt
        outputs['D_t_down'] = D_t
        outputs['L_pt_down'] = L_pt
        outputs['G_t_down'] = G_t
        outputs['EIRP_down'] = EIRP
        outputs['L_s_down'] = L_s
        outputs['G_rp_down'] = G_rp
        outputs['theta_r_down'] = theta_r
        outputs['L_pr_down'] = L_pr
        outputs['G_r_down'] = G_r
        outputs['Eb/No_down'] = Eb_No
        outputs['C/No_down'] = C_No
        outputs['Margin_down'] = Margin

"""
7. On-Board Computer and Data Handling Subsystem (OBCS): ARM Cortex-M
"""

@dataclass
class OBCSARMCortexM(om.ExplicitComponent):
    """Sizing of On-Board Computer and Data Handling Subsystem: ARM Cortex-M"""

    processor_speed: float
    num_cores: float
    memory_size: float
    num_data_channels: float

    # P_OBCS: float
    # m_OBCS: float
    # data_rate: float
    # memory_usage: float

    def __init__(self, processor_speed, num_cores, memory_size, num_data_channels):
        super().__init__()

        self.processor_speed = processor_speed
        self.num_cores = num_cores
        self.memory_size = memory_size
        self.num_data_channels = num_data_channels
        # self.P_OBCS = P_OBCS
        # self.m_OBCS = m_OBCS
        # self.data_rate = data_rate
        # self.memory_usage = memory_usage

    def setup(self):
        self.add_input('processor_speed', val=self.processor_speed, units="GHz")
        self.add_input('num_cores', val=self.num_cores)
        self.add_input('memory_size', val=self.memory_size)
        self.add_input('num_data_channels', val=self.num_data_channels)

        self.add_output('P_OBCS', units="W")
        self.add_output('m_OBCS', units="kg")
        self.add_output('data_rate')
        self.add_output('memory_usage')

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.processor_speed = inputs['processor_speed']
        self.num_cores = inputs['num_cores']
        self.memory_size = inputs['memory_size']
        self.num_data_channels = inputs['num_data_channels']

        # ARM Cortex-M characteristics (typical for embedded systems)
        base_power_per_core = 0.05  # Base power consumption per core (W) for ARM Cortex-M
        base_power_per_memory = 0.01  # Power per GB of memory (W)
        base_power_per_channel = 0.2  # Power consumption per data channel (W)
        max_data_rate_per_core = 100
        memory_usage_per_channel = 0.05
        power_per_core = base_power_per_core

        self.P_OBCS = (self.num_cores * power_per_core + self.memory_size * base_power_per_memory +
                       self.num_data_channels * base_power_per_channel)
        self.data_rate = self.processor_speed * self.num_cores * max_data_rate_per_core
        self.memory_usage = self.memory_size + self.num_data_channels * memory_usage_per_channel
        self.m_OBCS = 0.13  # fixed-value from https://www.endurosat.com/products/onboard-computer/

        outputs['P_OBCS'] = self.P_OBCS
        outputs['m_OBCS'] = self.m_OBCS
        outputs['data_rate'] = self.data_rate
        outputs['memory_usage'] = self.memory_usage