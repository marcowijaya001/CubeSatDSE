import openmdao.api as om
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

from MDO_Components import *

if __name__ == "__main__":

    prob = om.Problem()
    CubeSat = prob.model

    tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')

    CubeSat.add_subsystem('ThermalControl', om.Group(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.ThermalControl.add_subsystem('Radiator', ThermalControlRadiator(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_subsystem('Structure', om.Group(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.Structure.add_subsystem('Aluminum', StructureAluminum(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_subsystem('ADCS', om.Group(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.ADCS.add_subsystem('ReactionWheel', ReactionWheelNew(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.ADCS.add_subsystem('MomentumWheel', MomentumWheelNew(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_subsystem('Power', om.Group(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.Power.add_subsystem('SolarPanel', SolarPanel(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.Power.add_subsystem('SecondaryBattery', SecondaryBatteryNew(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_subsystem('Payload', om.Group(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.Payload.add_subsystem('RemoteSensing', PayloadRemoteSensing(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_subsystem('Communication', om.Group(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.Communication.add_subsystem('RadioTransceiver', CommunicationRadioTransceiver(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_subsystem('OBCS', om.Group(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.OBCS.add_subsystem('OBCSARMCortexM', OBCSARMCortexM(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_subsystem('MiscellaneousComponents', MiscellaneousComponents(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_subsystem('Aggregation', om.Group(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.Aggregation.add_subsystem('MassAggregation', MassAggregation(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.Aggregation.add_subsystem('PowerAggregation', PowerAggregation(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_design_var('Area_density', lower=1.5, upper=4.0)
    CubeSat.add_objective('M_total', scaler=1)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.setup()

    CubeSat.set_val('SolarConstant', val=tree.getroot().find('ThermalControl/SolarConstant').text)
    CubeSat.set_val('height_size', val=tree.getroot().find('Structure/height_size').text)
    CubeSat.set_val('edge_size', val=tree.getroot().find('Structure/edge_size').text)
    CubeSat.set_val('Op', val=tree.getroot().find('Constants/Mission/Op').text)
    CubeSat.set_val('Alt', val=tree.getroot().find('Constants/Mission/Alt').text)

    prob.run_driver()

    prob.model.list_inputs(val=True, units=True)
    prob.model.list_outputs(val=True, units=True)