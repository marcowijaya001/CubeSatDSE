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