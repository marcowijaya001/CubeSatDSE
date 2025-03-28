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