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