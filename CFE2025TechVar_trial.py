"""
Evaluation code for ADSG with component selection only
"""
from dataclasses import dataclass

import pandas as pd
from adore.optimization.api.factory_evaluator import *
from adore.api.schema import *
import matplotlib.pyplot as plt
from decorator import append
from matplotlib.lines import Line2D
import openmdao.api as om
import math
import numpy as np
import openmdao.func_api as omf
from pycparser.ply.ctokens import t_AND
from scipy.constants import metric_ton
import time

from scipy.io.matlab import MatReadWarning

from Components_for_CFE_techvar import *

class CubeSatTechVarCFE(ClassFactoryApiEvaluator):

    @staticmethod
    def get_class_factories() -> List[ClassFactory]:
        return[
            ClassFactory(
                el=ExternalComponentDef(name='Radiator Cu', auto=True),
                cls=ThermalControlRadiatorCu,
                props={
                    'sigma_radiator_Cu': ExternalQOIDef(name='sigma_radiator_Cu', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'alpha_radiator_Cu': ExternalQOIDef(name='alpha_radiator_Cu', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'epsilon_radiator_Cu': ExternalQOIDef(name='epsilon_radiator_Cu', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'beta_radiator_Cu': ExternalQOIDef(name='beta_radiator_Cu', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'SolarConstant_radiator_Cu': ExternalQOIDef(name='SolarConstant_radiator_Cu', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Q_int_Cu': ExternalQOIDef(name='Q_int_Cu', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'q_EarthIR_Cu': ExternalQOIDef(name='q_EarthIR_Cu', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'T_req_radiator_Cu': ExternalQOIDef(name='T_req_radiator_Cu', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    't_radiator_Cu': ExternalQOIDef(name='t_radiator_Cu', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'rho_radiator_Cu': ExternalQOIDef(name='rho_radiator_Cu', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Radiator Al', auto=True),
                cls=ThermalControlRadiatorAl,
                props={
                    'sigma_radiator_Al': ExternalQOIDef(name='sigma_radiator_Al', qoi_type=QOIType.INPUT_PARAM,
                                                        auto=True),
                    'alpha_radiator_Al': ExternalQOIDef(name='alpha_radiator_Al', qoi_type=QOIType.INPUT_PARAM,
                                                        auto=True),
                    'epsilon_radiator_Al': ExternalQOIDef(
                        name='epsilon_radiator_Al',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'beta_radiator_Al': ExternalQOIDef(name='beta_radiator_Al', qoi_type=QOIType.INPUT_PARAM,
                                                       auto=True),
                    'SolarConstant_radiator_Al': ExternalQOIDef(
                        name='SolarConstant_radiator_Al',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'Q_int_Al': ExternalQOIDef(name='Q_int_Al', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'q_EarthIR_Al': ExternalQOIDef(name='q_EarthIR_Al', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'T_req_radiator_Al': ExternalQOIDef(
                        name='T_req_radiator_Al',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    't_radiator_Al': ExternalQOIDef(name='t_radiator_Al', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'rho_radiator_Al': ExternalQOIDef(name='rho_radiator_Al', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Radiator CFRP', auto=True),
                cls=ThermalControlRadiatorCFRP,
                props={
                    'sigma_radiator_CFRP': ExternalQOIDef(
                        name='sigma_radiator_CFRP',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'alpha_radiator_CFRP': ExternalQOIDef(
                        name='alpha_radiator_CFRP',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'epsilon_radiator_CFRP': ExternalQOIDef(
                        name='epsilon_radiator_CFRP',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'beta_radiator_CFRP': ExternalQOIDef(
                        name='beta_radiator_CFRP',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'SolarConstant_radiator_CFRP': ExternalQOIDef(
                        name='SolarConstant_radiator_CFRP',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'Q_int_CFRP': ExternalQOIDef(
                        name='Q_int_CFRP',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'q_EarthIR_CFRP': ExternalQOIDef(
                        name='q_EarthIR_CFRP',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'T_req_radiator_CFRP': ExternalQOIDef(
                        name='T_req_radiator_CFRP',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    't_radiator_CFRP': ExternalQOIDef(
                        name='t_radiator_CFRP',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'rho_radiator_CFRP': ExternalQOIDef(
                        name='rho_radiator_CFRP',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Coating White', auto=True),
                cls=ThermalControlSurfaceFinishesWhite,
                props={
                    'sigma_coating_white': ExternalQOIDef(
                        name='sigma_coating_white',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'alpha_coating_white': ExternalQOIDef(
                        name='alpha_coating_white',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'epsilon_coating_white': ExternalQOIDef(
                        name='epsilon_coating_white',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'SolarConstant_coating_white': ExternalQOIDef(
                        name='SolarConstant_coating_white',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'edge_size_coating_white': ExternalQOIDef(
                        name='edge_size_coating_white',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'height_size_coating_white': ExternalQOIDef(
                        name='height_size_coating_white',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    't_coating_white': ExternalQOIDef(
                        name='t_coating_white',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'rho_coating_white': ExternalQOIDef(
                        name='rho_coating_white',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'T_req_coating_white': ExternalQOIDef(
                        name='T_req_coating_white',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Coating Black', auto=True),
                cls=ThermalControlSurfaceFinishesBlack,
                props={
                    'sigma_coating_black': ExternalQOIDef(
                        name='sigma_coating_black',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'alpha_coating_black': ExternalQOIDef(
                        name='alpha_coating_black',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'epsilon_coating_black': ExternalQOIDef(
                        name='epsilon_coating_black',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'SolarConstant_coating_black': ExternalQOIDef(
                        name='SolarConstant_coating_black',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'edge_size_coating_black': ExternalQOIDef(
                        name='edge_size_coating_black',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'height_size_coating_black': ExternalQOIDef(
                        name='height_size_coating_black',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    't_coating_black': ExternalQOIDef(
                        name='t_coating_black',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'rho_coating_black': ExternalQOIDef(
                        name='rho_coating_black',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'T_req_coating_black': ExternalQOIDef(
                        name='T_req_coating_black',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Coating Kapton', auto=True),
                cls=ThermalControlSurfaceFinishesKapton,
                props={
                    'sigma_coating_kapton': ExternalQOIDef(
                        name='sigma_coating_kapton',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'alpha_coating_kapton': ExternalQOIDef(
                        name='alpha_coating_kapton',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'epsilon_coating_kapton': ExternalQOIDef(
                        name='epsilon_coating_kapton',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'SolarConstant_coating_kapton': ExternalQOIDef(
                        name='SolarConstant_coating_kapton',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'edge_size_coating_kapton': ExternalQOIDef(
                        name='edge_size_coating_kapton',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'height_size_coating_kapton': ExternalQOIDef(
                        name='height_size_coating_kapton',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    't_coating_kapton': ExternalQOIDef(
                        name='t_coating_kapton',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'rho_coating_kapton': ExternalQOIDef(
                        name='rho_coating_kapton',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                    'T_req_coating_kapton': ExternalQOIDef(
                        name='T_req_coating_kapton',
                        qoi_type=QOIType.INPUT_PARAM,
                        auto=True
                    ),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Additive Manufacturing', auto=True),
                cls=StructureAdditiveManufacturing,
                props={
                    'g_am': ExternalQOIDef(name='g_am', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'rho_am': ExternalQOIDef(name='rho_am', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'tau_am': ExternalQOIDef(name='tau_am', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'safety_factor_am': ExternalQOIDef(name='safety_factor_am', qoi_type=QOIType.INPUT_PARAM,
                                                       auto=True),
                    'edge_size_am': ExternalQOIDef(name='edge_size_am', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'height_size_am': ExternalQOIDef(name='height_size_am', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'width_am': ExternalQOIDef(name='width_am', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Aluminum', auto=True),
                cls=StructureAluminum,
                props={
                    'g_aluminum': ExternalQOIDef(name='g_aluminum', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'rho_aluminum': ExternalQOIDef(name='rho_aluminum', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'tau_aluminum': ExternalQOIDef(name='tau_aluminum', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'safety_factor_aluminum': ExternalQOIDef(name='safety_factor_aluminum',
                                                             qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'width_aluminum': ExternalQOIDef(name='width_aluminum', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'height_size_aluminum': ExternalQOIDef(name='height_size_aluminum', qoi_type=QOIType.INPUT_PARAM,
                                                           auto=True),
                    'edge_size_aluminum': ExternalQOIDef(name='edge_size_aluminum', qoi_type=QOIType.INPUT_PARAM,
                                                         auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Reaction Wheel', auto=True),
                cls=ReactionWheel,
                props={
                    'Rwr': ExternalQOIDef(name='Rwr', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Rwav': ExternalQOIDef(name='Rwav', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Op_rw': ExternalQOIDef(name='Op_rw', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    't_slew': ExternalQOIDef(name='t_slew', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'MaxSA': ExternalQOIDef(name='MaxSA', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'edge_size_rw': ExternalQOIDef(name='edge_size_rw', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'height_size_rw': ExternalQOIDef(name='height_size_rw', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Momentum Wheel', auto=True),
                cls=MomentumWheel,
                props={
                    'Op_mw': ExternalQOIDef(name='Op_mw', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'T_total': ExternalQOIDef(name='T_total', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Mwr': ExternalQOIDef(name='Mwr', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Mwav': ExternalQOIDef(name='Mwav', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Maad': ExternalQOIDef(name='Maad', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'edge_size_mw': ExternalQOIDef(name='edge_size_mw', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'height_size_mw': ExternalQOIDef(name='height_size_mw', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Magnetorquers', auto=True),
                cls=Magnetorquers,
                props={
                    'T_total_mag': ExternalQOIDef(name='T_total_mag', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'B_mag': ExternalQOIDef(name='B_mag', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'I_mag': ExternalQOIDef(name='I_mag', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'A_mag': ExternalQOIDef(name='A_mag', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'r_mag': ExternalQOIDef(name='r_mag', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'rho_mag': ExternalQOIDef(name='rho_mag', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'd_mag': ExternalQOIDef(name='d_mag', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'rho_wire': ExternalQOIDef(name='rho_wire', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Solar Panel GaAs', auto=True),
                cls=SolarPanelGaAs,
                props={
                    'f_sunlight_GaAs': ExternalQOIDef(name='f_sunlight_GaAs', qoi_type=QOIType.DESIGN_VAR, auto=True),
                    'PDPY_GaAs': ExternalQOIDef(name='PDPY_GaAs', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'SF_GaAs': ExternalQOIDef(name='SF_GaAs', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Efficiency_sp_GaAs': ExternalQOIDef(name='Efficiency_sp_GaAs', qoi_type=QOIType.INPUT_PARAM,
                                                         auto=True),
                    'Wcsia_GaAs': ExternalQOIDef(name='Wcsia_GaAs', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Area_density_GaAs': ExternalQOIDef(name='Area_density_GaAs', qoi_type=QOIType.INPUT_PARAM,
                                                        auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Solar Panel Si', auto=True),
                cls=SolarPanelSi,
                props={
                    'f_sunlight_Si': ExternalQOIDef(name='f_sunlight_Si', qoi_type=QOIType.DESIGN_VAR, auto=True),
                    'PDPY_Si': ExternalQOIDef(name='PDPY_Si', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'SF_Si': ExternalQOIDef(name='SF_Si', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Efficiency_sp_Si': ExternalQOIDef(name='Efficiency_sp_Si', qoi_type=QOIType.INPUT_PARAM,
                                                       auto=True),
                    'Wcsia_Si': ExternalQOIDef(name='Wcsia_Si', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Area_density_Si': ExternalQOIDef(name='Area_density_Si', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Solar Panel MJ', auto=True),
                cls=SolarPanelMJ,
                props={
                    'f_sunlight_MJ': ExternalQOIDef(name='f_sunlight_MJ', qoi_type=QOIType.DESIGN_VAR, auto=True),
                    'PDPY_MJ': ExternalQOIDef(name='PDPY_MJ', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'SF_MJ': ExternalQOIDef(name='SF_MJ', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Efficiency_sp_MJ': ExternalQOIDef(name='Efficiency_sp_MJ', qoi_type=QOIType.INPUT_PARAM,
                                                       auto=True),
                    'Wcsia_MJ': ExternalQOIDef(name='Wcsia_MJ', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Area_density_MJ': ExternalQOIDef(name='Area_density_MJ', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Secondary Battery LiIon', auto=True),
                cls=SecondaryBatteryLiIon,
                props={
                    'MaxET_LiIon': ExternalQOIDef(name='MaxET_LiIon', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'DoD_LiIon': ExternalQOIDef(name='DoD_LiIon', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Ed_LiIon': ExternalQOIDef(name='Ed_LiIon', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Secondary Battery NiCd', auto=True),
                cls=SecondaryBatteryNiCd,
                props={
                    'MaxET_NiCd': ExternalQOIDef(name='MaxET_NiCd', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'DoD_NiCd': ExternalQOIDef(name='DoD_NiCd', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Ed_NiCd': ExternalQOIDef(name='Ed_NiCd', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Secondary Battery Na2S', auto=True),
                cls=SecondaryBatteryNa2S,
                props={
                    'MaxET_Na2S': ExternalQOIDef(name='MaxET_Na2S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'DoD_Na2S': ExternalQOIDef(name='DoD_Na2S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Ed_Na2S': ExternalQOIDef(name='Ed_Na2S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Primary Battery AgZn', auto=True),
                cls=PrimaryBatteryAgZn,
                props={
                    'MD_AgZn': ExternalQOIDef(name='MD_AgZn', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'e_pb_AgZn': ExternalQOIDef(name='e_pb_AgZn', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Primary Battery LiSOCl2', auto=True),
                cls=PrimaryBatteryLiSOCl2,
                props={
                    'MD_LiSOCl2': ExternalQOIDef(name='MD_LiSOCl2', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'e_pb_LiSOCl2': ExternalQOIDef(name='e_pb_LiSOCl2', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Primary Battery LiCl', auto=True),
                cls=PrimaryBatteryLiCl,
                props={
                    'MD_LiCl': ExternalQOIDef(name='MD_LiCl', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'e_pb_LiCl': ExternalQOIDef(name='e_pb_LiCl', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Remote Sensing', auto=True),
                cls=PayloadRemoteSensingNew,
                props={
                    'x_optical': ExternalQOIDef(name='x_optical', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Alt_payload': ExternalQOIDef(name='Alt_payload', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    #'GSD_optical': ExternalQOIDef(name='GSD_optical', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'lambda_optical': ExternalQOIDef(name='lambda_optical', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Q_optical': ExternalQOIDef(name='Q_optical', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Radio Transceiver UHF', auto=True),
                cls=CommunicationRadioTransceiverUHF,
                props={
                    'f_down_UHF': ExternalQOIDef(name='f_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'P_t_down_UHF': ExternalQOIDef(name='P_t_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_l_down_UHF': ExternalQOIDef(name='L_l_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'theta_t_down_UHF': ExternalQOIDef(name='theta_t_down_UHF', qoi_type=QOIType.INPUT_PARAM,
                                                       auto=True),
                    'e_t_down_UHF': ExternalQOIDef(name='e_t_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'S_down_UHF': ExternalQOIDef(name='S_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_a_down_UHF': ExternalQOIDef(name='L_a_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'eff_down_UHF': ExternalQOIDef(name='eff_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'D_r_down_UHF': ExternalQOIDef(name='D_r_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'e_r_down_UHF': ExternalQOIDef(name='e_r_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'T_s_down_UHF': ExternalQOIDef(name='T_s_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'R_down_UHF': ExternalQOIDef(name='R_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'BER_down_UHF': ExternalQOIDef(name='BER_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_imp_down_UHF': ExternalQOIDef(name='L_imp_down_UHF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Eb_No_req_down_UHF': ExternalQOIDef(name='Eb_No_req_down_UHF', qoi_type=QOIType.INPUT_PARAM,
                                                         auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Radio Transceiver S', auto=True),
                cls=CommunicationRadioTransceiverS,
                props={
                    'f_down_S': ExternalQOIDef(name='f_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'P_t_down_S': ExternalQOIDef(name='P_t_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_l_down_S': ExternalQOIDef(name='L_l_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'theta_t_down_S': ExternalQOIDef(name='theta_t_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'e_t_down_S': ExternalQOIDef(name='e_t_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'S_down_S': ExternalQOIDef(name='S_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_a_down_S': ExternalQOIDef(name='L_a_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'eff_down_S': ExternalQOIDef(name='eff_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'D_r_down_S': ExternalQOIDef(name='D_r_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'e_r_down_S': ExternalQOIDef(name='e_r_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'T_s_down_S': ExternalQOIDef(name='T_s_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'R_down_S': ExternalQOIDef(name='R_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'BER_down_S': ExternalQOIDef(name='BER_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_imp_down_S': ExternalQOIDef(name='L_imp_down_S', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Eb_No_req_down_S': ExternalQOIDef(name='Eb_No_req_down_S', qoi_type=QOIType.INPUT_PARAM,
                                                       auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Radio Transceiver L', auto=True),
                cls=CommunicationRadioTransceiverL,
                props={
                    'f_down_L': ExternalQOIDef(name='f_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'P_t_down_L': ExternalQOIDef(name='P_t_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_l_down_L': ExternalQOIDef(name='L_l_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'theta_t_down_L': ExternalQOIDef(name='theta_t_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'e_t_down_L': ExternalQOIDef(name='e_t_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'S_down_L': ExternalQOIDef(name='S_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_a_down_L': ExternalQOIDef(name='L_a_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'eff_down_L': ExternalQOIDef(name='eff_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'D_r_down_L': ExternalQOIDef(name='D_r_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'e_r_down_L': ExternalQOIDef(name='e_r_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'T_s_down_L': ExternalQOIDef(name='T_s_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'R_down_L': ExternalQOIDef(name='R_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'BER_down_L': ExternalQOIDef(name='BER_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_imp_down_L': ExternalQOIDef(name='L_imp_down_L', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Eb_No_req_down_L': ExternalQOIDef(name='Eb_No_req_down_L', qoi_type=QOIType.INPUT_PARAM,
                                                       auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='OBCSARMCortexM', auto=True),
                cls=OBCSARMCortexM,
                props={
                    'processor_speed': ExternalQOIDef(name='processor_speed', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'num_cores': ExternalQOIDef(name='num_cores', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'memory_size': ExternalQOIDef(name='memory_size', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'num_data_channels': ExternalQOIDef(name='num_data_channels', qoi_type=QOIType.INPUT_PARAM, auto=True),
                }
            ),
        ]

    @staticmethod
    def get_metrics_factory() -> MetricsFactory:
        return MetricsFactory(metrics={
            'M_total': ExternalQOIDef(name='M_total', qoi_type=QOIType.OBJECTIVE, auto=True),
            'data_downloaded': ExternalQOIDef(name='data_downloaded', qoi_type=QOIType.OBJECTIVE, auto=True),
            'P_total': ExternalQOIDef(name='P_total', qoi_type=QOIType.OBJECTIVE, auto=True),
            'M_thermal_coating': ExternalQOIDef(name='M_thermal_coating', qoi_type=QOIType.METRIC, auto=True),
            'A_coating': ExternalQOIDef(name='A_coating', qoi_type=QOIType.METRIC, auto=True),
            'A_radiator': ExternalQOIDef(name='A_radiator', qoi_type=QOIType.METRIC, auto=True),
            'M_thermal_radiator': ExternalQOIDef(name='M_thermal_radiator', qoi_type=QOIType.METRIC, auto=True),
            't_structure_aluminum': ExternalQOIDef(name='t_structure_aluminum', qoi_type=QOIType.METRIC, auto=True),
            'M_structure_aluminum': ExternalQOIDef(name='M_structure_aluminum', qoi_type=QOIType.METRIC, auto=True),
            't_structure_am': ExternalQOIDef(name='t_structure_am', qoi_type=QOIType.METRIC, auto=True),
            'M_structure_am': ExternalQOIDef(name='M_structure_am', qoi_type=QOIType.METRIC, auto=True),
            'h_size_rw': ExternalQOIDef(name='h_size_rw', qoi_type=QOIType.METRIC, auto=True),
            'Mrw': ExternalQOIDef(name='Mrw', qoi_type=QOIType.METRIC, auto=True),
            'Prw': ExternalQOIDef(name='Prw', qoi_type=QOIType.METRIC, auto=True),
            'Mram': ExternalQOIDef(name='Mram', qoi_type=QOIType.METRIC, auto=True),
            'Mmw': ExternalQOIDef(name='Mmw', qoi_type=QOIType.METRIC, auto=True),
            'Pmw': ExternalQOIDef(name='Pmw', qoi_type=QOIType.METRIC, auto=True),
            'm_mag': ExternalQOIDef(name='m_mag', qoi_type=QOIType.METRIC, auto=True),
            'P_mag': ExternalQOIDef(name='P_mag', qoi_type=QOIType.METRIC, auto=True),
            'P_generated': ExternalQOIDef(name='P_generated', qoi_type=QOIType.METRIC, auto=True),
            'A_solarpanel': ExternalQOIDef(name='A_solarpanel', qoi_type=QOIType.METRIC, auto=True),
            'm_solarpanel': ExternalQOIDef(name='m_solarpanel', qoi_type=QOIType.METRIC, auto=True),
            'E_battery_sb': ExternalQOIDef(name='E_battery_sb', qoi_type=QOIType.METRIC, auto=True),
            'Msb': ExternalQOIDef(name='Msb', qoi_type=QOIType.METRIC, auto=True),
            'E_battery_pb': ExternalQOIDef(name='E_battery_pb', qoi_type=QOIType.METRIC, auto=True),
            'Mpb': ExternalQOIDef(name='Mpb', qoi_type=QOIType.METRIC, auto=True),
            'f_optical': ExternalQOIDef(name='f_optical', qoi_type=QOIType.METRIC, auto=True),
            'D_optical': ExternalQOIDef(name='D_optical', qoi_type=QOIType.METRIC, auto=True),
            'm_payload': ExternalQOIDef(name='m_payload', qoi_type=QOIType.METRIC, auto=True),
            'P_payload': ExternalQOIDef(name='P_payload', qoi_type=QOIType.METRIC, auto=True),
            'l_payload': ExternalQOIDef(name='l_payload', qoi_type=QOIType.METRIC, auto=True),
            'w_payload': ExternalQOIDef(name='w_payload', qoi_type=QOIType.METRIC, auto=True),
            'h_payload': ExternalQOIDef(name='h_payload', qoi_type=QOIType.METRIC, auto=True),
            'M_comm_down': ExternalQOIDef(name='M_comm_down', qoi_type=QOIType.METRIC, auto=True),
            'P_comm_down': ExternalQOIDef(name='P_comm_down', qoi_type=QOIType.METRIC, auto=True),
            'P_OBCS': ExternalQOIDef(name='P_OBCS', qoi_type=QOIType.METRIC, auto=True),
            'm_OBCS': ExternalQOIDef(name='m_OBCS', qoi_type=QOIType.METRIC, auto=True),
            'data_rate': ExternalQOIDef(name='data_rate', qoi_type=QOIType.METRIC, auto=True),
            'memory_usage': ExternalQOIDef(name='memory_usage', qoi_type=QOIType.METRIC, auto=True),
        })

    def _evaluate(self, architecture: Architecture, arch_qois: List[ArchQOI], **kwargs) -> Dict[ArchQOI, float]:

        radiator_cu = self.instantiate(architecture, factories='Radiator Cu')
        radiator_al = self.instantiate(architecture, factories='Radiator Al')
        radiator_cfrp = self.instantiate(architecture, factories='Radiator CFRP')
        coating_white = self.instantiate(architecture, factories='Coating White')
        coating_black = self.instantiate(architecture, factories='Coating Black')
        coating_kapton = self.instantiate(architecture, factories='Coating Kapton')
        aluminum = self.instantiate(architecture, factories='Aluminum')
        additive_manufacturing = self.instantiate(architecture, factories='Additive Manufacturing')
        reaction_wheel = self.instantiate(architecture, factories='Reaction Wheel')
        momentum_wheel = self.instantiate(architecture, factories='Momentum Wheel')
        magnetorquers = self.instantiate(architecture, factories='Magnetorquers')
        solar_panel_gaas = self.instantiate(architecture, factories='Solar Panel GaAs')
        solar_panel_si = self.instantiate(architecture, factories='Solar Panel Si')
        solar_panel_mj = self.instantiate(architecture, factories='Solar Panel MJ')
        secondary_battery_liion = self.instantiate(architecture, factories='Secondary Battery LiIon')
        secondary_battery_nicd = self.instantiate(architecture, factories='Secondary Battery NiCd')
        secondary_battery_na2s = self.instantiate(architecture, factories='Secondary Battery Na2S')
        primary_battery_agzn = self.instantiate(architecture, factories='Primary Battery AgZn')
        primary_battery_lisocl2 = self.instantiate(architecture, factories='Primary Battery LiSOCl2')
        primary_battery_licl = self.instantiate(architecture, factories='Primary Battery LiCl')
        radio_transceiver_uhf = self.instantiate(architecture, factories='Radio Transceiver UHF')
        radio_transceiver_s = self.instantiate(architecture, factories='Radio Transceiver S')
        radio_transceiver_l = self.instantiate(architecture, factories='Radio Transceiver L')
        remote_sensing = self.instantiate(architecture, factories='Remote Sensing')
        armcortexm = self.instantiate(architecture, factories='OBCSARMCortexM')

        #CubeSat = om.Group()

        prob = om.Problem()
        CubeSat = prob.model

        # Creating discipline of OpenMDAO from instantiated components
        components = [('RadiatorCu', radiator_cu), ('RadiatorAl', radiator_al), ('RadiatorCFRP', radiator_cfrp),
                      ('CoatingWhite', coating_white), ('CoatingBlack', coating_black), ('CoatingKapton', coating_kapton),
                      ('Aluminum', aluminum), ('AdditiveManufacturing', additive_manufacturing),
                      ('ReactionWheel', reaction_wheel), ('MomentumWheel', momentum_wheel), ('Magnetorquers', magnetorquers),
                      ('SolarPanelGaAs', solar_panel_gaas), ('SolarPanelSi', solar_panel_si), ('SolarPanelMJ', solar_panel_mj),
                      ('SecondaryBatteryLiIon', secondary_battery_liion), ('SecondaryBatteryNiCd', secondary_battery_nicd), ('SecondaryBatteryNa2S', secondary_battery_na2s),
                      ('PrimaryBatteryAgZn', primary_battery_agzn), ('PrimaryBatteryLiSOCl2', primary_battery_lisocl2), ('PrimaryBatteryLiCl', primary_battery_licl),
                      ('RadioTransceiverUHF', radio_transceiver_uhf), ('RadioTransceiverS', radio_transceiver_s), ('RadioTransceiverL', radio_transceiver_l),
                      ('RemoteSensing', remote_sensing), ('OBCSARMCortexM', armcortexm)]

        for name, component in components:
            if len(component) != 0:
                CubeSat.add_subsystem(name, component[0], promotes_inputs=['*'], promotes_outputs=['*'])


        # Mass Aggregation
        def MassAggregation(m_payload):

            # Mass of Thermal Control Subsystem
            mass_thermal = [(coating_white, 'CoatingWhite.M_thermal_coating'), (coating_black, 'CoatingBlack.M_thermal_coating'),
                            (coating_kapton, 'CoatingKapton.M_thermal_coating'), (radiator_al, 'RadiatorAl.M_thermal_radiator'),
                            (radiator_cu, 'RadiatorCu.M_thermal_radiator'), (radiator_cfrp, 'RadiatorCFRP.M_thermal_radiator')]

            for component, value in mass_thermal:
                if len(component) != 0:
                    M_thermal = prob.get_val(value)

            # Mass of Structure Subsystem
            mass_structure = [(aluminum, 'Aluminum.M_structure_aluminum'), (additive_manufacturing, 'AdditiveManufacturing.M_structure_am')]

            for component, value in mass_structure:
                if len(component) != 0:
                    M_structure = prob.get_val(value)

            # Mass of Communication Subsystem
            mass_communication = [(radio_transceiver_uhf, 'RadioTransceiverUHF.M_comm_down'), (radio_transceiver_s, 'RadioTransceiverS.M_comm_down'),
                                  (radio_transceiver_l,'RadioTransceiverL.M_comm_down')]

            for component, value in mass_communication:
                if len(component) != 0:
                    M_communication = prob.get_val(value)

            # Mass of Power Subsystem
            mass_power = [(primary_battery_agzn, 'PrimaryBatteryAgZn.Mpb'), (primary_battery_lisocl2, 'PrimaryBatteryLiSOCl2.Mpb'), (primary_battery_licl, 'PrimaryBatteryLiCl.Mpb'),
                          (solar_panel_gaas, 'SolarPanelGaAs.m_solarpanel'), (solar_panel_si, 'SolarPanelSi.m_solarpanel'), (solar_panel_mj, 'SolarPanelMJ.m_solarpanel'),
                          (secondary_battery_liion, 'SecondaryBatteryLiIon.Msb'), (secondary_battery_nicd, 'SecondaryBatteryNiCd.Msb'), (secondary_battery_na2s, 'SecondaryBatteryNa2S.Msb')]

            m_power = []
            for component, value in mass_power:
                if len(component) != 0:
                    m_power.append(prob.get_val(value))
                M_power = sum(m_power)

            # Mass of Attitude Determination and Control Subsystem
            mass_ADCS = [(momentum_wheel, 'MomentumWheel.Mmw'), (reaction_wheel, 'ReactionWheel.Mrw'), (magnetorquers, 'Magnetorquers.m_mag')]

            m_ADCS = []
            for component, value in mass_ADCS:
                if len(component) != 0:
                    m_ADCS.append(prob.get_val(value))
                M_ADCS = sum(m_ADCS)

            #M_payload = prob.get_val('RemoteSensing.m_payload')
            M_OBCS = prob.get_val('OBCSARMCortexM.m_OBCS')

            M_antenna = 0.081
            M_sensor = 0.035
            M_cable = 0.1
            M_misc = M_antenna + M_sensor + M_cable

            M_total = (m_payload + M_thermal + M_structure + M_ADCS + M_power + M_communication + M_OBCS + M_misc) + (0.1*np.random.uniform(-1,1))

            return M_total

        f = omf.wrap(MassAggregation).declare_partials(['*'], ['*']).add_input('m_payload', units="kg").add_output('M_total', units="kg")
        CubeSat.add_subsystem('MassAggregation', om.ExplicitFuncComp(f), promotes_inputs=['*'], promotes_outputs=['*'])

        #CubeSat.connect('m_payload', 'M_payload')

        # Power Aggregation
        def PowerAggregation():

            # Power of Attitude Determination and Control Subsystem
            power_ADCS = [(momentum_wheel, 'MomentumWheel.Pmw'), (reaction_wheel, 'ReactionWheel.Prw'),
                         (magnetorquers, 'Magnetorquers.P_mag')]

            p_ADCS = []
            for component, value in power_ADCS:
                if len(component) != 0:
                    p_ADCS.append(prob.get_val(value))
                P_ADCS = sum(p_ADCS)

            # Power of Communication Subsystem
            power_communication = [(radio_transceiver_uhf, 'RadioTransceiverUHF.P_comm_down'), (radio_transceiver_s, 'RadioTransceiverS.P_comm_down'),
                                  (radio_transceiver_l,'RadioTransceiverL.P_comm_down')]

            for component, value in power_communication:
                if len(component) != 0:
                    P_communication = prob.get_val(value)

            P_payload = prob.get_val('RemoteSensing.P_payload')
            P_OBCS = prob.get_val('OBCSARMCortexM.P_OBCS')

            P_antenna = 0.05
            P_sensor = 0.315
            P_misc = P_antenna + P_sensor

            P_total = P_ADCS + P_payload + P_communication + P_OBCS + P_misc
            return P_total

        g = omf.wrap(PowerAggregation).add_output('P_total', units="W")
        CubeSat.add_subsystem('PowerAggregation', om.ExplicitFuncComp(g), promotes_inputs=['*'], promotes_outputs=['*'])

        # Multidisciplinary Optimization (MDO) without optimization iteration recording
        prob = om.Problem(CubeSat)

        # A. Design variables of subsystem
        CubeSat.add_design_var('GSD_optical', lower=20, upper=40) # 1. Ground Sampling Distance design variable
        # 2. Load factor design variable
        if len(aluminum) != 0:
            CubeSat.add_design_var('g_aluminum', lower=80, upper=100)
        else:
            CubeSat.add_design_var('g_am', lower=80, upper=100)
        # 3. Operational temperature design variable
        T_required = [(coating_white, 'T_req_coating_white'), (coating_black, 'T_req_coating_black'),
                        (coating_kapton, 'T_req_coating_kapton'), (radiator_al, 'T_req_radiator_Al'),
                        (radiator_cu, 'T_req_radiator_Cu'), (radiator_cfrp, 'T_req_radiator_CFRP')]
        for component, T_req in T_required:
            if len(component) != 0:
                CubeSat.add_design_var(T_req, lower=283, upper=300)
        # 4. Communication frequency design variable
        if len(radio_transceiver_uhf) != 0:
            CubeSat.add_design_var('f_down_UHF', lower=0.5, upper=1.0)
        elif len(radio_transceiver_l) != 0:
            CubeSat.add_design_var('f_down_L', lower=1.0, upper=2.0)
        else:
            CubeSat.add_design_var('f_down_S', lower=2.0, upper=4.0)
        # 5. ADCS radius (momentum wheel, reaction wheel, magnetorquers) design variables
        if len(reaction_wheel) != 0:
            CubeSat.add_design_var('Rwr', lower=0.02, upper=0.03)
            CubeSat.add_design_var('Rwav', lower=650, upper=700)
        if len(momentum_wheel) != 0:
            CubeSat.add_design_var('Mwr', lower=0.04, upper=0.05)
            CubeSat.add_design_var('Mwav', lower=600, upper=650)
        if len(magnetorquers) != 0:
            CubeSat.add_design_var('r_mag', lower=0.04, upper=0.05)
        """
        # B. Mission requirements as OpenMDAO design variables: maximum eclipse time
        CubeSat.add_design_var('Alt_payload', lower=500, upper=700)  # 1. Altitude
        # 2. Maximum eclipse time; 3. Proportion of sunlight; 4. Mission duration
        mission_reqs = [(secondary_battery_liion, 'MaxET_LiIon', 30, 40), (secondary_battery_nicd, 'MaxET_NiCd', 30, 40), (secondary_battery_na2s, 'MaxET_Na2S', 30, 40),
                        (solar_panel_si, 'f_sunlight_Si', 0.2, 0.8), (solar_panel_gaas, 'f_sunlight_GaAs', 0.2, 0.8), (solar_panel_mj, 'f_sunlight_MJ', 0.2, 0.8),
                        (primary_battery_licl, 'MD_LiCl', 0.8, 1.2), (primary_battery_agzn, 'MD_AgZn', 0.8, 1.2), (primary_battery_lisocl2, 'MD_LiSOCl2', 0.8, 1.2)]
        for component, req, lower, upper in mission_reqs:
            if len(component) != 0:
                CubeSat.add_design_var(req, lower=lower, upper=upper)
        """
        CubeSat.add_objective('M_total', scaler=1) # Objective to minimize CubeSat total mass
        CubeSat.add_constraint('l_payload', upper=0.1) # Constraint to limit the dimension of payload
        CubeSat.add_constraint('w_payload', upper=0.1) # Constraint to limit the dimension of payload
        CubeSat.add_constraint('h_payload', upper=0.1) # Constraint to limit the dimension of payload
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.options['tol'] = 1e-4
        prob.driver.options['maxiter'] = 500
        prob.setup()
        prob.run_driver()
        """
        # Multidisciplinary Optimization (MDO) with optimization iteration recording
        prob = om.Problem(CubeSat)
        CubeSat.add_design_var('Alt_payload', lower=500, upper=600) # Altitude requirement
        CubeSat.add_design_var('GSD_optical', lower=20, upper=40) # Ground Sampling Distance requirement
        # Load factor requirement
        if len(aluminum) != 0:
            CubeSat.add_design_var('g_aluminum', lower=80, upper=100)
        else:
            CubeSat.add_design_var('g_am', lower=80, upper=100)
        # Operational temperature requirement
        T_required = [(coating_white, 'T_req_coating_white'), (coating_black, 'T_req_coating_black'),
                        (coating_kapton, 'T_req_coating_kapton'), (radiator_al, 'T_req_radiator_Al'),
                        (radiator_cu, 'T_req_radiator_Cu'), (radiator_cfrp, 'T_req_radiator_CFRP')]
        for component, T_req in T_required:
            if len(component) != 0:
                CubeSat.add_design_var(T_req, lower=283, upper=300)
        CubeSat.add_objective('M_total', scaler=1) # Objective to minimize CubeSat total mass
        CubeSat.add_constraint('l_payload', upper=0.1) # Constraint to limit the dimension of payload
        CubeSat.add_constraint('w_payload', upper=0.1) # Constraint to limit the dimension of payload
        CubeSat.add_constraint('h_payload', upper=0.1) # Constraint to limit the dimension of payload
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'COBYLA'
        prob.driver.options['tol'] = 1e-4
        # Create a recorder variable
        recorder = om.SqliteRecorder('cases.sql')
        # Attach a recorder to the problem
        prob.add_recorder(recorder)
        prob.driver.add_recorder(recorder)
        prob.setup()
        prob.set_solver_print(0)
        prob.run_driver()
        prob.record("final_state")
        prob.cleanup()
        # Instantiate your CaseReader
        cr = om.CaseReader(prob.get_outputs_dir() / "cases.sql")
        # Get the first case from the recorder
        driver_cases = cr.get_cases('driver', recurse=False)
        DV1 = []
        DV2 = []
        DV3 = []
        DV4 = []
        OBJ = []
        for case in driver_cases:
            DV1.append(case['GSD_optical'])
            DV2.append(case['Alt_payload'])
            if len(aluminum) != 0:
                DV3.append(case['g_aluminum'])
            else:
                DV3.append(case['g_am'])
            T_required = [(coating_white, 'T_req_coating_white'), (coating_black, 'T_req_coating_black'),
                          (coating_kapton, 'T_req_coating_kapton'), (radiator_al, 'T_req_radiator_Al'),
                          (radiator_cu, 'T_req_radiator_Cu'), (radiator_cfrp, 'T_req_radiator_CFRP')]
            for component, T_req in T_required:
                if len(component) != 0:
                    DV4.append(case[T_req])
            OBJ.append(case['M_total'])
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
        ax1.plot(np.arange(len(DV1)), np.array(DV1))
        ax1.set(ylabel='GSD', title='Optimization History')
        ax1.grid()
        ax2.plot(np.arange(len(DV2)), np.array(DV2))
        ax2.set(ylabel='Altitude')
        ax2.grid()
        ax3.plot(np.arange(len(DV3)), np.array(DV3))
        ax3.set(ylabel='Load factor')
        ax3.grid()
        ax4.plot(np.arange(len(DV4)), np.array(DV4))
        ax4.set(ylabel='Temperature')
        ax4.grid()
        ax5.plot(np.arange(len(OBJ)), np.array(OBJ))
        ax5.set(xlabel='Iterations', ylabel='CubeSat Mass')
        ax5.grid()
        plt.show()
        
        # Multidisciplinary Analysis (MDA)
        prob = om.Problem(CubeSat)
        prob.setup()
        prob.run_model()
        """
        # Conditional statement to return the QOIs of selected components

        ## Thermal control subsystem
        if len(radiator_cu) or len(radiator_al) or len(radiator_cfrp) != 0:
            prop_radiator = [(radiator_cu, 'RadiatorCu.M_thermal_radiator', 'RadiatorCu.A_radiator'),
                             (radiator_al, 'RadiatorAl.M_thermal_radiator', 'RadiatorAl.A_radiator'),
                             (radiator_cfrp, 'RadiatorCFRP.M_thermal_radiator', 'RadiatorCFRP.A_radiator')]
            for component, mass, area in prop_radiator:
                if len(component) != 0:
                    M_thermal_radiator = prob.get_val(mass)
                    A_radiator = prob.get_val(area)
                    M_thermal_coating = 0
                    A_coating = 0
        else:
            prop_coating = [(coating_white, 'CoatingWhite.M_thermal_coating', 'CoatingWhite.A_coating'),
                            (coating_black, 'CoatingBlack.M_thermal_coating', 'CoatingBlack.A_coating'),
                            (coating_kapton, 'CoatingKapton.M_thermal_coating', 'CoatingKapton.A_coating')]
            for component, mass, area in prop_coating:
                if len(component) != 0:
                    M_thermal_coating = prob.get_val(mass)
                    A_coating = prob.get_val(area)
                    M_thermal_radiator = 0
                    A_radiator = 0

        ## Communication subsystem
        prop_radio = [(radio_transceiver_uhf, 'RadioTransceiverUHF.M_comm_down', 'RadioTransceiverUHF.P_comm_down', 'RadioTransceiverUHF.data_downloaded'),
                      (radio_transceiver_l, 'RadioTransceiverL.M_comm_down', 'RadioTransceiverL.P_comm_down', 'RadioTransceiverL.data_downloaded'),
                      (radio_transceiver_s, 'RadioTransceiverS.M_comm_down', 'RadioTransceiverS.P_comm_down', 'RadioTransceiverS.data_downloaded')]
        for component, mass, power, data in prop_radio:
            if len(component) != 0:
                M_comm_down = prob.get_val(mass)
                P_comm_down = prob.get_val(power)
                data_downloaded = prob.get_val(data)

        ## Structure subsystem
        if len(aluminum) != 0:
            t_aluminum = prob.get_val('Aluminum.t_structure_aluminum')
            M_aluminum = prob.get_val('Aluminum.M_structure_aluminum')
            t_am = 0
            M_am = 0
        elif len(additive_manufacturing) != 0:
            t_aluminum = 0
            M_aluminum = 0
            t_am = prob.get_val('AdditiveManufacturing.t_structure_am')
            M_am = prob.get_val('AdditiveManufacturing.M_structure_am')

        ## Power subsystem
        if len(solar_panel_gaas) or len(solar_panel_si) or len(solar_panel_mj) != 0:
            prop_sp = [(solar_panel_gaas, 'SolarPanelGaAs.P_generated', 'SolarPanelGaAs.m_solarpanel', 'SolarPanelGaAs.A_solarpanel'),
                       (solar_panel_si, 'SolarPanelSi.P_generated', 'SolarPanelSi.m_solarpanel', 'SolarPanelSi.A_solarpanel'),
                       (solar_panel_mj, 'SolarPanelMJ.P_generated', 'SolarPanelMJ.m_solarpanel', 'SolarPanelMJ.A_solarpanel')]
            for component, power, mass, area in prop_sp:
                if len(component) != 0:
                    Mpb = 0
                    E_battery_pb = 0
                    P_generated = prob.get_val(power)
                    m_solarpanel = prob.get_val(mass)
                    A_solarpanel = prob.get_val(area)
            prop_sb = [(secondary_battery_liion, 'SecondaryBatteryLiIon.Msb', 'SecondaryBatteryLiIon.E_battery_sb'),
                       (secondary_battery_nicd, 'SecondaryBatteryNiCd.Msb', 'SecondaryBatteryNiCd.E_battery_sb'),
                       (secondary_battery_na2s, 'SecondaryBatteryNa2S.Msb', 'SecondaryBatteryNa2S.E_battery_sb')]
            if len(secondary_battery_liion) or len(secondary_battery_nicd) or len(secondary_battery_na2s) != 0:
                for component, mass, energy in prop_sb:
                    if len(component) != 0:
                        Msb = prob.get_val(mass)
                        E_battery_sb = prob.get_val(energy)
            else:
                Msb = 0
                E_battery_sb = 0
        elif len(primary_battery_agzn) or primary_battery_lisocl2 or primary_battery_licl != 0:
            prop_pb = [(primary_battery_agzn, 'PrimaryBatteryAgZn.Mpb', 'PrimaryBatteryAgZn.E_battery_pb'),
                       (primary_battery_lisocl2, 'PrimaryBatteryLiSOCl2.Mpb', 'PrimaryBatteryLiSOCl2.E_battery_pb'),
                       (primary_battery_licl, 'PrimaryBatteryLiCl.Mpb', 'PrimaryBatteryLiCl.E_battery_pb')]
            for component, mass, energy in prop_pb:
                if len(component) != 0:
                    Mpb = prob.get_val(mass)
                    E_battery_pb = prob.get_val(energy)
                    P_generated = 0
                    m_solarpanel = 0
                    A_solarpanel = 0
                    Msb = 0
                    E_battery_sb = 0

        ## ADCS
        if len(momentum_wheel) and len(reaction_wheel) and len(magnetorquers) != 0:
            Mmw = prob.get_val('MomentumWheel.Mmw')
            Pmw = prob.get_val('MomentumWheel.Pmw')
            Mram = prob.get_val('MomentumWheel.Mram')
            Mrw = prob.get_val('ReactionWheel.Mrw')
            Prw = prob.get_val('ReactionWheel.Prw')
            h_size_rw = prob.get_val('ReactionWheel.h_size_rw')
            m_mag = prob.get_val('Magnetorquers.m_mag')
            P_mag = prob.get_val('Magnetorquers.P_mag')
        else:
            if len(momentum_wheel) != 0:
                Mmw = prob.get_val('MomentumWheel.Mmw')
                Pmw = prob.get_val('MomentumWheel.Pmw')
                Mram = prob.get_val('MomentumWheel.Mram')
                Mrw = 0
                Prw = 0
                h_size_rw = 0
                m_mag = 0
                P_mag = 0
            elif len(reaction_wheel) != 0:
                Mmw = 0
                Pmw = 0
                Mram = 0
                Mrw = prob.get_val('ReactionWheel.Mrw')
                Prw = prob.get_val('ReactionWheel.Prw')
                h_size_rw = prob.get_val('ReactionWheel.h_size_rw')
                m_mag = 0
                P_mag = 0
            elif len(magnetorquers) != 0:
                Mmw = 0
                Pmw = 0
                Mram = 0
                Mrw = 0
                Prw = 0
                h_size_rw = 0
                m_mag = prob.get_val('Magnetorquers.m_mag')
                P_mag = prob.get_val('Magnetorquers.P_mag')

        results = {
            'M_total': np.atleast_1d(prob.get_val('MassAggregation.M_total')),
            'P_total': np.atleast_1d(prob.get_val('PowerAggregation.P_total')),
            'M_thermal_coating': M_thermal_coating,
            'A_coating': A_coating,
            'M_thermal_radiator': M_thermal_radiator,
            'A_radiator': A_radiator,
            't_structure_aluminum': t_aluminum,
            'M_structure_aluminum': M_aluminum,
            't_structure_am': t_am,
            'M_structure_am': M_am,
            'h_size_rw': h_size_rw,
            'Mrw': Mrw,
            'Prw': Prw,
            'Mram': Mram,
            'Mmw': Mmw,
            'Pmw': Pmw,
            'm_mag': m_mag,
            'P_mag': P_mag,
            'P_generated': P_generated,
            'A_solarpanel': A_solarpanel,
            'm_solarpanel': m_solarpanel,
            'E_battery_sb': E_battery_sb,
            'Msb': Msb,
            'E_battery_pb': E_battery_pb,
            'Mpb': Mpb,
            'f_optical': np.atleast_1d(prob.get_val('RemoteSensing.f_optical')),
            'D_optical': np.atleast_1d(prob.get_val('RemoteSensing.D_optical')),
            'm_payload': np.atleast_1d(prob.get_val('RemoteSensing.m_payload')),
            'P_payload': np.atleast_1d(prob.get_val('RemoteSensing.P_payload')),
            'l_payload': np.atleast_1d(prob.get_val('RemoteSensing.l_payload')),
            'w_payload': np.atleast_1d(prob.get_val('RemoteSensing.w_payload')),
            'h_payload': np.atleast_1d(prob.get_val('RemoteSensing.h_payload')),
            'M_comm_down': M_comm_down,
            'P_comm_down': P_comm_down,
            'data_downloaded': data_downloaded,
            'P_OBCS': np.atleast_1d(prob.get_val('OBCSARMCortexM.P_OBCS')),
            'm_OBCS': np.atleast_1d(prob.get_val('OBCSARMCortexM.m_OBCS')),
            'data_rate': np.atleast_1d(prob.get_val('OBCSARMCortexM.data_rate')),
            'memory_usage': np.atleast_1d(prob.get_val('OBCSARMCortexM.memory_usage')),
        }
        #prob.model.RemoteSensing.list_vars(val=True, units=True)
        #prob.model.MassAggregation.list_vars(val=True, units=True)

        return self.process_results(architecture, arch_qois, results)


evaluator = CubeSatTechVarCFE.from_file('CubeSat_TechVar_components_selection_20Mar.adore') # ADORE model for components selection only, doesn't include change of requirements
evaluator.update_external_database()
evaluator.to_file('CubeSat_TechVar_components_selection_linked.adore')

start_time = time.time()
# Post-processing 1
dv1 = []
dv2 = []
dv3 = []
dv4 = []
obj1 = []
obj2 =[]
obj3 = []

for _ in range(200000):
    architecture, dv, is_active = evaluator.get_architecture(evaluator.get_random_design_vector())
    #design_vector = (2, 0, 5, 0, 0, 1, 0, 2, 0, 0)
    #architecture, dv, is_active = evaluator.get_architecture(design_vector)
    obj = evaluator.evaluate(architecture)
    print(f'DV {dv!r} --> OBJ {obj!r}')

    dv1.append(dv[1])
    dv2.append(dv[7]) # GSD
    dv3.append(dv[6]) # f_sunlight
    #dv4.append(dv[8]) # mission duration
    obj1.append(obj[0][2]) # power
    obj2.append(obj[0][1]) # mass
    obj3.append(obj[0][0]) # data downloaded

end_time = time.time()
simulation_time = end_time - start_time
print(f"Simulation Time: {simulation_time:.2f} seconds")

plt.figure()

# Post-processing 4:
plt.title('')
plt.xlabel('Total Mass of CubeSat [kg]')
plt.ylabel('Power Consumption [W]')
plt.scatter(obj2, obj1, s=10)
plt.show()

evaluator.to_file('CubeSat_TechVar_components_selection_evaluated.adore')
evaluator.save_results_csv('CubeSat_TechVar.components_selection_evaluated.csv')