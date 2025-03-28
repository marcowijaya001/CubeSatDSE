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