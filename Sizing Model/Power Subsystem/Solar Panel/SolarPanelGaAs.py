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