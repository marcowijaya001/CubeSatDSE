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
from pycparser.ply.ctokens import t_AND
from scipy.constants import metric_ton
import time

from Sizing.Components_for_CFE import *

class CubeSatClassFactoryEvaluator(ClassFactoryApiEvaluator):

    @staticmethod
    def get_class_factories() -> List[ClassFactory]:
        return[
            ClassFactory(
                el=ExternalComponentDef(name='Radiator', auto=True),
                cls=ThermalControlRadiator,
                props={
                    'sigma_radiator': ExternalQOIDef(name='sigma_radiator', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'alpha_radiator': ExternalQOIDef(name='alpha_radiator', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'epsilon_radiator': ExternalQOIDef(name='epsilon_radiator', qoi_type=QOIType.INPUT_PARAM,
                                                       auto=True),
                    'beta_radiator': ExternalQOIDef(name='beta_radiator', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'SolarConstant_radiator': ExternalQOIDef(name='SolarConstant_radiator',
                                                             qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Q_int': ExternalQOIDef(name='Q_int', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'q_EarthIR': ExternalQOIDef(name='q_EarthIR', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'T_req_radiator': ExternalQOIDef(name='T_req_radiator', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    't_radiator': ExternalQOIDef(name='t_radiator', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'rho_radiator': ExternalQOIDef(name='rho_radiator', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    #'A_radiator': ExternalQOIDef(name='A_radiator', qoi_type=QOIType.METRIC, auto=True),
                    #'M_thermal_radiator': ExternalQOIDef(name='M_thermal_radiator', qoi_type=QOIType.METRIC, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Coating', auto=True),
                cls=ThermalControlSurfaceFinishes,
                props={
                    'sigma_coating': ExternalQOIDef(name='sigma_coating', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'alpha_coating': ExternalQOIDef(name='alpha_coating', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'epsilon_coating': ExternalQOIDef(name='epsilon_coating', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'SolarConstant_coating': ExternalQOIDef(name='SolarConstant_coating', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'edge_size_coating': ExternalQOIDef(name='edge_size_coating', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'height_size_coating': ExternalQOIDef(name='height_size_coating', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    't_coating': ExternalQOIDef(name='t_coating', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'rho_coating': ExternalQOIDef(name='rho_coating', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'T_req_coating': ExternalQOIDef(name='T_req_coating', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    #'M_thermal_coating': ExternalQOIDef(name='M_thermal_coating', qoi_type=QOIType.METRIC, auto=True),
                    #'A_coating': ExternalQOIDef(name='A_coating', qoi_type=QOIType.METRIC, auto=True),
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
                    # 't_structure_am': ExternalQOIDef(name='', qoi_type=QOIType.METRIC, auto=True),
                    # 'M_structure_am': ExternalQOIDef(name='', qoi_type=QOIType.METRIC, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Aluminum', auto=True),
                cls=StructureAluminum,
                props={
                    'g_aluminum': ExternalQOIDef(name='g_aluminum', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'rho_aluminum': ExternalQOIDef(name='rho_aluminum', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'tau_aluminum': ExternalQOIDef(name='tau_aluminum', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'safety_factor_aluminum': ExternalQOIDef(name='safety_factor_aluminum', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'width_aluminum': ExternalQOIDef(name='width_aluminum', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'height_size_aluminum': ExternalQOIDef(name='height_size_aluminum', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'edge_size_aluminum': ExternalQOIDef(name='edge_size_aluminum', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    #'t_structure_aluminum': ExternalQOIDef(name='t_structure_aluminum', qoi_type=QOIType.METRIC, auto=True),
                    #'M_structure_aluminum': ExternalQOIDef(name='M_structure_aluminum', qoi_type=QOIType.METRIC, auto=True),
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
                    #'h_size_rw': ExternalQOIDef(name='h_size_rw', qoi_type=QOIType.METRIC, auto=True),
                    #'Mrw': ExternalQOIDef(name='Mrw', qoi_type=QOIType.METRIC, auto=True),
                    #'Prw': ExternalQOIDef(name='Prw', qoi_type=QOIType.METRIC, auto=True),
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
                    #'Mram': ExternalQOIDef(name='Mram', qoi_type=QOIType.METRIC, auto=True),
                    #'Mmw': ExternalQOIDef(name='Mmw', qoi_type=QOIType.METRIC, auto=True),
                    #'Pmw': ExternalQOIDef(name='Pmw', qoi_type=QOIType.METRIC, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Solar Panel', auto=True),
                cls=SolarPanel,
                props={
                    'f_sunlight': ExternalQOIDef(name='f_sunlight', qoi_type=QOIType.DESIGN_VAR, auto=True),
                    'PDPY': ExternalQOIDef(name='PDPY', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'SF': ExternalQOIDef(name='SF', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Efficiency_sp': ExternalQOIDef(name='Efficiency_sp', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Wcsia': ExternalQOIDef(name='Wcsia', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Area_density': ExternalQOIDef(name='Area_density', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    #'P_generated': ExternalQOIDef(name='P_generated', qoi_type=QOIType.METRIC, auto=True),
                    #'A_solarpanel': ExternalQOIDef(name='A_solarpanel', qoi_type=QOIType.METRIC, auto=True),
                    #'m_solarpanel': ExternalQOIDef(name='m_solarpanel', qoi_type=QOIType.METRIC, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Secondary Battery', auto=True),
                cls=SecondaryBattery,
                props={
                    'MaxET': ExternalQOIDef(name='MaxET', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'DoD': ExternalQOIDef(name='DoD', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Ed': ExternalQOIDef(name='Ed', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    #'E_battery_sb': ExternalQOIDef(name='E_battery_sb', qoi_type=QOIType.METRIC, auto=True),
                    #'Msb': ExternalQOIDef(name='Msb', qoi_type=QOIType.METRIC, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Primary Battery', auto=True),
                cls=PrimaryBattery,
                props={
                    'MD': ExternalQOIDef(name='MD', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'e_pb': ExternalQOIDef(name='e_pb', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    #'E_battery_pb': ExternalQOIDef(name='E_battery_pb', qoi_type=QOIType.METRIC, auto=True),
                    #'Mpb': ExternalQOIDef(name='Mpb', qoi_type=QOIType.METRIC, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Remote Sensing', auto=True),
                cls=PayloadRemoteSensing,
                props={
                    'x_optical': ExternalQOIDef(name='x_optical', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Alt_payload': ExternalQOIDef(name='Alt_payload', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'GSD_optical': ExternalQOIDef(name='GSD_optical', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'lambda_optical': ExternalQOIDef(name='lambda_optical', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Q_optical': ExternalQOIDef(name='Q_optical', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    #'f_optical': ExternalQOIDef(name='f_optical', qoi_type=QOIType.METRIC, auto=True),
                    #'D_optical': ExternalQOIDef(name='D_optical', qoi_type=QOIType.METRIC, auto=True),
                    #'m_payload': ExternalQOIDef(name='m_payload', qoi_type=QOIType.METRIC, auto=True),
                    #'P_payload': ExternalQOIDef(name='P_payload', qoi_type=QOIType.METRIC, auto=True),
                    #'l_payload': ExternalQOIDef(name='l_payload', qoi_type=QOIType.METRIC, auto=True),
                    #'w_payload': ExternalQOIDef(name='w_payload', qoi_type=QOIType.METRIC, auto=True),
                    #'h_payload': ExternalQOIDef(name='h_payload', qoi_type=QOIType.METRIC, auto=True),
                }
            ),
            ClassFactory(
                el=ExternalComponentDef(name='Radio Transceiver', auto=True),
                cls=CommunicationRadioTransceiver,
                props={
                    'f_down': ExternalQOIDef(name='f_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'P_t_down': ExternalQOIDef(name='P_t_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_l_down': ExternalQOIDef(name='L_l_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'theta_t_down': ExternalQOIDef(name='theta_t_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'e_t_down': ExternalQOIDef(name='e_t_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'S_down': ExternalQOIDef(name='S_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_a_down': ExternalQOIDef(name='L_a_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'eff_down': ExternalQOIDef(name='eff_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'D_r_down': ExternalQOIDef(name='D_r_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'e_r_down': ExternalQOIDef(name='e_r_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'T_s_down': ExternalQOIDef(name='T_s_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'R_down': ExternalQOIDef(name='R_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'BER_down': ExternalQOIDef(name='BER_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'L_imp_down': ExternalQOIDef(name='L_imp_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    'Eb_No_req_down': ExternalQOIDef(name='Eb_No_req_down', qoi_type=QOIType.INPUT_PARAM, auto=True),
                    #'M_comm_down': ExternalQOIDef(name='M_comm_down', qoi_type=QOIType.METRIC, auto=True),
                    #'P_comm_down': ExternalQOIDef(name='P_comm_down', qoi_type=QOIType.METRIC, auto=True),
                    #'Br_down': ExternalQOIDef(name='Br_down', qoi_type=QOIType.METRIC, auto=True),
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
                    #'P_OBCS': ExternalQOIDef(name='P_OBCS', qoi_type=QOIType.METRIC, auto=True),
                    #'m_OBCS': ExternalQOIDef(name='m_OBCS', qoi_type=QOIType.METRIC, auto=True),
                    #'data_rate': ExternalQOIDef(name='data_rate', qoi_type=QOIType.METRIC, auto=True),
                    #'memory_usage': ExternalQOIDef(name='memory_usage', qoi_type=QOIType.METRIC, auto=True),
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

        coating = self.instantiate(architecture, factories='Coating')
        radiator = self.instantiate(architecture, factories='Radiator')
        aluminum = self.instantiate(architecture, factories='Aluminum')
        additive_manufacturing = self.instantiate(architecture, factories='Additive Manufacturing')
        reaction_wheel = self.instantiate(architecture, factories='Reaction Wheel')
        momentum_wheel = self.instantiate(architecture, factories='Momentum Wheel')
        solar_panel = self.instantiate(architecture, factories='Solar Panel')
        secondary_battery = self.instantiate(architecture, factories='Secondary Battery')
        primary_battery = self.instantiate(architecture, factories='Primary Battery')
        remote_sensing = self.instantiate(architecture, factories='Remote Sensing')
        radio_transceiver = self.instantiate(architecture, factories='Radio Transceiver')
        armcortexm = self.instantiate(architecture, factories='OBCSARMCortexM')

        CubeSat = om.Group()

        # 1. Thermal Control Subsystem
        if len(coating) != 0:
            CubeSat.add_subsystem('Coating', coating[0])
        elif len(radiator) != 0:
            CubeSat.add_subsystem('Radiator', radiator[0])

        # 2. Structure Subsystem
        if len(aluminum) != 0:
            CubeSat.add_subsystem('Aluminum', aluminum[0])
        elif len(additive_manufacturing) != 0:
            CubeSat.add_subsystem('AdditiveManufacturing', additive_manufacturing[0])

        #  3. Attitude Determination and Control Subsystem
        if len(reaction_wheel) != 0:
            CubeSat.add_subsystem('ReactionWheel', reaction_wheel[0])
        if len(momentum_wheel) != 0:
            CubeSat.add_subsystem('MomentumWheel', momentum_wheel[0])

        # 4. Power Subsystem
        if len(solar_panel) != 0:
            CubeSat.add_subsystem('SolarPanel', solar_panel[0])
        if len(secondary_battery) != 0:
            CubeSat.add_subsystem('SecondaryBattery', secondary_battery[0])
        if len(primary_battery) != 0:
            CubeSat.add_subsystem('PrimaryBattery', primary_battery[0])

        # 5. Payload Subsystem
        CubeSat.add_subsystem('RemoteSensing', remote_sensing[0])

        # 6. Communication Subsystem
        CubeSat.add_subsystem('RadioTransceiver', radio_transceiver[0])

        # 7. On-Board Computer Subsystem
        CubeSat.add_subsystem('OBCSARMCortexM', armcortexm[0])

        # 8. Mass Aggregation
        def MassAggregation():

            # Mass of Thermal Control Subsystem
            if len(coating) != 0:
                M_thermal = prob.get_val('Coating.M_thermal_coating')
            else:
                M_thermal = prob.get_val('Radiator.M_thermal_radiator')

            # Mass of Structure Subsystem
            if len(aluminum) != 0:
                M_structure = prob.get_val('Aluminum.M_structure_aluminum')
            else:
                M_structure = prob.get_val('AdditiveManufacturing.M_structure_am')

            # Mass of Attitude Determination and Control Subsystem
            if len(reaction_wheel) != 0 and len(momentum_wheel) != 0:
                M_ADCS = prob.get_val('ReactionWheel.Mrw') + prob.get_val('MomentumWheel.Mmw')
            elif len(reaction_wheel) != 0 and len(momentum_wheel) == 0:
                M_ADCS = prob.get_val('ReactionWheel.Mrw')
            elif len(reaction_wheel) == 0 and len(momentum_wheel) != 0:
                M_ADCS = prob.get_val('MomentumWheel.Mmw')

            # Mass of Power Subsystem
            if len(solar_panel) != 0 and len(secondary_battery) != 0:
                M_power = prob.get_val('SolarPanel.m_solarpanel') + prob.get_val('SecondaryBattery.Msb')
            elif len(solar_panel) != 0 and len(secondary_battery) == 0:
                M_power = prob.get_val('SolarPanel.m_solarpanel')
            elif len(primary_battery) != 0:
                M_power = prob.get_val('PrimaryBattery.Mpb')

            M_payload = prob.get_val('RemoteSensing.m_payload')
            M_communication = prob.get_val('RadioTransceiver.M_comm_down')
            M_OBCS = prob.get_val('OBCSARMCortexM.m_OBCS')

            M_antenna = 0.081
            M_sensor = 0.035
            M_cable = 0.1
            M_misc = M_antenna + M_sensor + M_cable

            M_total = M_thermal + M_structure + M_ADCS + M_power + M_payload + M_communication + M_OBCS + M_misc

            return M_total

        f = omf.wrap(MassAggregation).add_output('M_total', units="kg")
        CubeSat.add_subsystem('MassAggregation', om.ExplicitFuncComp(f))

        # 9. Power Aggregation
        def PowerAggregation():

            # Power of Attitude Determination and Control Subsystem
            if len(reaction_wheel) != 0 and len(momentum_wheel) != 0:
                P_ADCS = prob.get_val('ReactionWheel.Prw') + prob.get_val('MomentumWheel.Pmw')
            elif len(reaction_wheel) != 0 and len(momentum_wheel) == 0:
                P_ADCS = prob.get_val('ReactionWheel.Prw')
            elif len(reaction_wheel) == 0 and len(momentum_wheel) != 0:
                P_ADCS = prob.get_val('MomentumWheel.Pmw')

            P_payload = prob.get_val('RemoteSensing.P_payload')
            P_communication = prob.get_val('RadioTransceiver.P_comm_down')
            P_OBCS = prob.get_val('OBCSARMCortexM.P_OBCS')

            P_antenna = 0.05
            P_sensor = 0.315
            P_misc = P_antenna + P_sensor

            P_total = P_ADCS + P_payload + P_communication + P_OBCS + P_misc

            return P_total

        g = omf.wrap(PowerAggregation).add_output('P_total', units="W")
        CubeSat.add_subsystem('PowerAggregation', om.ExplicitFuncComp(g))

        """
        # Multidisciplinary Optimization (MDO)
        prob = om.Problem(CubeSat)
        CubeSat.add_design_var('RemoteSensing.GSD_optical', lower=28, upper=30)
        CubeSat.add_objective('MassAggregation.M_total', scaler=1)
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-6
        prob.setup()
        prob.run_driver()
        """

        # Multidisciplinary Analysis (MDA)
        prob = om.Problem(CubeSat)
        prob.setup()
        prob.run_model()

        # Conditional statement to return the QOIs of selected components
        if len(coating) != 0:
            M_coating = prob.get_val('Coating.M_thermal_coating')
            A_coating = prob.get_val('Coating.A_coating')
            M_radiator = 0
            A_radiator = 0
        elif len(radiator) != 0:
            M_coating = 0
            A_coating = 0
            M_radiator = prob.get_val('Radiator.M_thermal_radiator')
            A_radiator = prob.get_val('Radiator.A_radiator')

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

        if len(reaction_wheel) != 0 and len(momentum_wheel) != 0:
            h_size_rw = prob.get_val('ReactionWheel.h_size_rw')
            Mrw = prob.get_val('ReactionWheel.Mrw')
            Prw = prob.get_val('ReactionWheel.Prw')
            Mram = prob.get_val('MomentumWheel.Mram')
            Mmw = prob.get_val('MomentumWheel.Mmw')
            Pmw = prob.get_val('MomentumWheel.Pmw')
        elif len(reaction_wheel) != 0 and len(momentum_wheel) == 0:
            h_size_rw = prob.get_val('ReactionWheel.h_size_rw')
            Mrw = prob.get_val('ReactionWheel.Mrw')
            Prw = prob.get_val('ReactionWheel.Prw')
            Mram = 0
            Mmw = 0
            Pmw = 0
        elif len(reaction_wheel) == 0 and len(momentum_wheel) != 0:
            h_size_rw = 0
            Mrw = 0
            Prw = 0
            Mram = prob.get_val('MomentumWheel.Mram')
            Mmw = prob.get_val('MomentumWheel.Mmw')
            Pmw = prob.get_val('MomentumWheel.Pmw')

        if len(solar_panel) != 0 and len(secondary_battery) != 0:
            P_generated = prob.get_val('SolarPanel.P_generated')
            A_solarpanel = prob.get_val('SolarPanel.A_solarpanel')
            m_solarpanel = prob.get_val('SolarPanel.m_solarpanel')
            E_battery_sb = prob.get_val('SecondaryBattery.E_battery_sb')
            Msb = prob.get_val('SecondaryBattery.Msb')
            E_battery_pb = 0
            Mpb = 0
        elif len(solar_panel) != 0 and len(secondary_battery) == 0:
            P_generated = prob.get_val('SolarPanel.P_generated')
            A_solarpanel = prob.get_val('SolarPanel.A_solarpanel')
            m_solarpanel = prob.get_val('SolarPanel.m_solarpanel')
            E_battery_sb = 0
            Msb = 0
            E_battery_pb = 0
            Mpb = 0
        elif len(primary_battery) != 0:
            P_generated = 0
            A_solarpanel = 0
            m_solarpanel = 0
            E_battery_sb = 0
            Msb = 0
            E_battery_pb = prob.get_val('PrimaryBattery.E_battery_pb')
            Mpb = prob.get_val('PrimaryBattery.Mpb')

        results = {
            'M_total': np.atleast_1d(prob.get_val('MassAggregation.M_total')),
            'P_total': np.atleast_1d(prob.get_val('PowerAggregation.P_total')),
            'M_thermal_coating': M_coating,
            'A_coating': A_coating,
            'A_radiator': A_radiator,
            'M_thermal_radiator': M_radiator,
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
            'M_comm_down': np.atleast_1d(prob.get_val('RadioTransceiver.M_comm_down')),
            'P_comm_down': np.atleast_1d(prob.get_val('RadioTransceiver.P_comm_down')),
            'data_downloaded': np.atleast_1d(prob.get_val('RadioTransceiver.data_downloaded')),
            'P_OBCS': np.atleast_1d(prob.get_val('OBCSARMCortexM.P_OBCS')),
            'm_OBCS': np.atleast_1d(prob.get_val('OBCSARMCortexM.m_OBCS')),
            'data_rate': np.atleast_1d(prob.get_val('OBCSARMCortexM.data_rate')),
            'memory_usage': np.atleast_1d(prob.get_val('OBCSARMCortexM.memory_usage')),
        }
        #prob.model.list_outputs(val=True, units=True)
        return self.process_results(architecture, arch_qois, results)


#evaluator = CubeSatClassFactoryEvaluator.from_file('CubeSat_System_metricoutside.adore')
evaluator = CubeSatClassFactoryEvaluator.from_file('CubeSat_System_27Jan_1.adore')
#evaluator = CubeSatClassFactoryEvaluator.from_file('CubeSat_System_31Jan.adore')
evaluator.update_external_database()
evaluator.to_file('CubeSat_System_linked.adore')
"""
for _ in range(150):
    architecture, dv, is_active = evaluator.get_architecture(evaluator.get_random_design_vector())
    obj = evaluator.evaluate(architecture)
    print(f'DV {dv!r} --> OBJ {obj!r}')
"""
start_time = time.time()
# Post-processing 1
dv1 = []
dv2 = []
dv3 = []
dv4 = []
obj1 = []
obj2 =[]
obj3 = []

for _ in range(200):
    architecture, dv, is_active = evaluator.get_architecture(evaluator.get_random_design_vector())
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

"""
# Post-processing 1:
plt.title('Power vs Mass for different power architecture')
colors = {0: 'b', 1: 'g', 2: 'y', 3: 'r'}
for i in range(len(dv3)):
    color = colors[dv3[i]]
    plt.scatter(obj2[i], obj1[i], color=color, s=10, label=f'Color {dv1[i]}')
plt.xlabel('Total Mass of CubeSat [kg]')
plt.ylabel('Power Consumption [W]')
architecture = {0:'f_sunlight 20%', 1:'f_sunlight 40%', 2:'f_sunlight 60%',
                3:'f_sunlight 80%'}
legend_elements = [Line2D([0], [0], marker='o', color='w', label=architecture[i], markerfacecolor=colors[i], markersize=5) for i in range (4)]
plt.legend(handles=legend_elements)
plt.show()

# Post-processing 2: 
plt.title('Total Mass vs Ground Sampling Distance (GSD)')
plt.xlabel('Ground Sampling Distance (GSD) requirement [m]')
plt.ylabel('Total Mass of CubeSat [kg]')
plt.scatter(dv2, obj2, s=10)
plt.show()

# Post-processing 3:
plt.title('Power vs Mass for different Ground Sampling Distance (GSD) requirement')
colors = {0: 'r', 1: 'b', 2: 'g'}
for i in range(len(dv2)):
    color = colors[dv2[i]]
    plt.scatter(obj2[i], obj1[i], color=color, s=10, label=f'Color {dv1[i]}')
plt.xlabel('Total Mass of CubeSat [kg]')
plt.ylabel('Power Consumption [W]')
architecture = {0:'GSD 20 [m]', 1:'GSD 30 [m]', 2:'GSD 40 [m]'}
legend_elements = [Line2D([0], [0], marker='o', color='w', label=architecture[i], markerfacecolor=colors[i], markersize=5) for i in range (3)]
plt.legend(handles=legend_elements)
plt.show()
"""
# Post-processing 4:
plt.title('')
plt.xlabel('Total Mass of CubeSat [kg]')
plt.ylabel('Power Consumption [W]')
plt.scatter(obj2, obj1, s=10)
plt.show()
"""
# Post-processing 5:
plt.title('Power vs Mass for different mission duration requirement')
colors = {0: 'g', 1: 'b', 2: 'r'}
for i in range(len(dv4)):
    color = colors[dv4[i]]
    plt.scatter(obj2[i], obj1[i], color=color, s=10, label=f'Color {dv1[i]}')
plt.xlabel('Total Mass of CubeSat [kg]')
plt.ylabel('Power Consumption [W]')
architecture = {0:'mission duration 1 day', 1:'mission duration 1 week', 2:'mission duration 1 month'}
legend_elements = [Line2D([0], [0], marker='o', color='w', label=architecture[i], markerfacecolor=colors[i], markersize=5) for i in range (3)]
plt.legend(handles=legend_elements)
plt.show()

# Post-processing 6:
plt.title('Power vs Mass for different power architecture')
colors = {0: 'g', 1: 'b'}
for i in range(len(dv1)):
    color = colors[dv1[i]]
    plt.scatter(obj2[i], obj1[i], color=color, s=10, label=f'Color {dv1[i]}')
plt.xlabel('Total Mass of CubeSat [kg]')
plt.ylabel('Power Consumption [W]')
architecture = {0:'by fixed energy capacity (battery only)', 1:'by harnessing energy during mission (solar panel + battery)'}
legend_elements = [Line2D([0], [0], marker='o', color='w', label=architecture[i], markerfacecolor=colors[i], markersize=5) for i in range (2)]
plt.legend(handles=legend_elements)
plt.show()

# Post-processing 7:
plt.xlabel('Mission Duration')
plt.ylabel('Total Mass of CubeSat [kg]')
plt.scatter(dv4, obj2, s=10)
architecture = {0:'1 day', 1:'1 week', 2:'1 month'}
plt.show()
"""
evaluator.to_file('CubeSat_System_evaluated.adore')
evaluator.save_results_csv('CubeSat_System_class_factory_evaluator.csv')

