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