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