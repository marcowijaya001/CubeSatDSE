import scipy
import openmdao.api as om
import math
import numpy as np
import xml.etree.ElementTree as ET

"""
MDO Components as Sizing Models of CubeSat (1-6), Mass and Power Aggregation, and MDO wrapper 
"""

"""
1. Thermal Control Subsystem: Surface Finishes/Coating and Radiator
"""

class ThermalControlSurfaceFinishes(om.ExplicitComponent):
    """Surface finishes as passive thermal control sizing"""
    """
    1. Possible requirement is operating temperature, to see after calculation if it is within the operational range 
    for each component. For example, to check whether a material with absorptivity and emissivity exists to satisfy the 
    thermal requirement
    2. Another possible checking requirement is geometry/size. To fulfill certain temperature requirement, how much area 
    should be covered by the coatings  
    """

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        # Inputs
        self.add_input('sigma', val=varval('ThermalControl/sigma'), units="W/m**2/K**4")
        self.add_input('alpha', val=varval('ThermalControl/SurfaceFinishes/alpha'))
        self.add_input('epsilon', val=varval('ThermalControl/SurfaceFinishes/epsilon'))
        self.add_input('SolarConstant', val=varval('ThermalControl/SolarConstant'), units="W/m**2")
        self.add_input('edge_size', val=varval('Structure/edge_size'), units="cm")
        self.add_input('height_size', val=varval('Structure/height_size'), units="cm")
        self.add_input('t_coating', val=varval('ThermalControl/t_coating'), units="mm")
        self.add_input('rho_coating', val=varval('ThermalControl/rho_coating'), units="kg/m**2")
        self.add_input('T_req', val=varval('ThermalControl/T_req'), units="K")

        # Outputs
        #self.add_output('T_operating', units="C")
        self.add_output('M_thermal', units="kg") #shared variable
        self.add_output('P_thermal', units="W")
        self.add_output('Ap', units="m**2")

        # Every output depends on several inputs
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        sigma = inputs['sigma']
        alpha = inputs['alpha']
        epsilon = inputs['epsilon']
        SolarConstant = inputs['SolarConstant']
        edge_size = inputs['edge_size']
        height_size = inputs['height_size']
        t_coating = inputs['t_coating']
        rho_coating = inputs['rho_coating']
        T_req = inputs['T_req']

        #Ap = edge_size*height_size
        A = 4*edge_size*height_size + 2*edge_size**2
        #T_operating = (((alpha/epsilon) * SolarConstant * (Ap/A)) / sigma)**0.25
        #M_thermal = rho_coating * (A/10000) * (t_coating/1000)
        P_thermal = 0.0

        Ap = (sigma * T_req**4 * A/10000) / (alpha * SolarConstant / epsilon)
        M_thermal = rho_coating * Ap * (t_coating*0.001)

        #outputs['T_operating'] = T_operating-273.15
        outputs['M_thermal'] = M_thermal
        outputs['P_thermal'] = P_thermal
        outputs['Ap'] = Ap

class ThermalControlRadiator(om.ExplicitComponent):
    """Radiator sizing"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        # Inputs
        self.add_input('sigma', val=varval('ThermalControl/sigma'))
        self.add_input('alpha', val=varval('ThermalControl/Radiator/alpha'))
        self.add_input('epsilon', val=varval('ThermalControl/Radiator/epsilon'))
        self.add_input('beta', val=varval('ThermalControl/beta'))
        self.add_input('SolarConstant', units="W/m**2")
        self.add_input('Q_int', val=varval('ThermalControl/Q_int'), units="W")
        self.add_input('q_EarthIR', val=varval('ThermalControl/q_EarthIR'), units="W/m**2")
        self.add_input('T_req', val=varval('ThermalControl/T_req'), units="K")
        self.add_input('t_radiator', val=varval('ThermalControl/t_radiator'), units="mm")
        self.add_input('rho_radiator', val=varval('ThermalControl/rho_radiator'), units="kg/m**3")

        # Outputs
        self.add_output('A_rad', units="m**2")
        self.add_output('M_thermal', units="kg") #shared variable
        self.add_output('P_thermal', units="W")

        # Every output depends on several inputs
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        sigma = inputs['sigma']
        alpha = inputs['alpha']
        epsilon = inputs['epsilon']
        beta = inputs['beta']
        SolarConstant = inputs['SolarConstant']
        Q_int = inputs['Q_int']
        q_EarthIR = inputs['q_EarthIR']
        T_req = inputs['T_req']
        t_radiator = inputs['t_radiator']
        rho_radiator = inputs['rho_radiator']

        q_ext = alpha * (SolarConstant + beta*SolarConstant + q_EarthIR)
        q_rad = epsilon * sigma * T_req**4
        A_rad = Q_int / (q_rad - q_ext)
        M_thermal = A_rad * (t_radiator*0.001) * rho_radiator
        P_thermal = 0.0

        outputs['A_rad'] = A_rad
        outputs['M_thermal'] = M_thermal
        outputs['P_thermal'] = P_thermal


"""
2. Structure Subsystem: Aluminum and Additive Manufacturing
"""

class StructureAluminum(om.ExplicitComponent):
    """Structural sizing of CubeSat using aluminum as material"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        # Inputs
        self.add_input('M_total', units="kg") #shared variable
        self.add_input('g', val=varval('Structure/g'), units="m/s**2")
        self.add_input('rho_aluminum', val=varval('Structure/Material/Aluminum/rho_aluminum'), units="kg/m**3")
        self.add_input('tau_aluminum', val=varval('Structure/Material/Aluminum/tau_aluminum'), units="N/m**2")
        self.add_input('safety_factor', val=varval('Structure/safety_factor'))
        self.add_input('width', val=varval('Structure/width'), units="cm")
        self.add_input('height_size', units="cm")
        self.add_input('edge_size', units="cm")

        # Outputs
        self.add_output('t_structure', units="mm")
        self.add_output('M_structure', units="kg") #shared variable

        # Every output depends on several inputs
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        M_approx_total = inputs['M_total']
        g = inputs['g']
        rho_aluminum = inputs['rho_aluminum']
        tau_aluminum = inputs['tau_aluminum']
        safety_factor = inputs['safety_factor']
        width = inputs['width']
        height_size = inputs['height_size']
        edge_size = inputs['edge_size']

        F = M_approx_total * g * safety_factor
        P = width * 0.01
        t_structure = F / (P * tau_aluminum)
        A_structure = (4*(edge_size*0.01)*(height_size*0.01) + 2*(edge_size*0.01)**2)*0.5
        M_structure = rho_aluminum*A_structure*t_structure

        outputs['t_structure'] = t_structure
        outputs['M_structure'] = M_structure

class StructureAdditiveManufacturing(om.ExplicitComponent):
    """Structural sizing of CubeSat using additive manufacturing's materials"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        # Inputs
        self.add_input('CubeSatFactor', val=varval('Structure/CubeSatFactor'))
        self.add_input('M_total', units="kg") #shared variable
        self.add_input('g', val=varval('Structure/g'), units="m/s**2")
        self.add_input('rho_am', val=varval('Structure/Material/AdditiveManufacturing/rho_am'), units="kg/m**3")
        self.add_input('tau_am', val=varval('Structure/Material/AdditiveManufacturing/tau_am'), units="N/m**2")
        self.add_input('safety_factor', val=varval('Structure/safety_factor'))
        self.add_input('edge_size', val=varval('Structure/edge_size'), units="cm")
        self.add_input('height_size', val=varval('Structure/height_size'), units="cm")
        self.add_input('width', val=varval('Structure/width'), units="cm")

        # Outputs
        self.add_output('t_structure', units="mm")
        self.add_output('M_structure', units="kg") #shared variable

        # Every output depends on several inputs
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        CubeSatFactor = inputs['CubeSatFactor']
        M_approx_total = inputs['M_total']
        g = inputs['g']
        rho_am = inputs['rho_am']
        tau_am = inputs['tau_am']
        safety_factor = inputs['safety_factor']
        width = inputs['width']
        height_size = inputs['height_size']
        edge_size = inputs['edge_size']

        F = M_approx_total * g * safety_factor
        P = width * 0.01
        t_structure = F / (P * tau_am)
        A_structure = (4 * (edge_size * 0.01) * (height_size * 0.01) + 2 * (edge_size * 0.01) ** 2)*0.5
        M_structure = rho_am * A_structure * t_structure

        outputs['t_structure'] = t_structure
        outputs['M_structure'] = M_structure


"""
3. Attitude Determination and Control Subsystem (ADCS): Reaction Wheel and Momentum Wheel
"""

class ReactionWheelNew(om.ExplicitComponent):
    """Sizing of reaction wheel"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        self.add_input('M_total', units="kg") #shared variable
        self.add_input('Rwr', val=varval('ADCS/Reaction_wheel/Rwr'), units="m")
        self.add_input('Rwav', val=varval('ADCS/Reaction_wheel/Rwav'), units="rad/s")
        self.add_input('Op', units="min")
        self.add_input('t_slew', val=varval('Constants/Mission/t_slew'), units="s")
        self.add_input('MaxSA', val=varval('Constants/Mission/MaxSA'), units="deg")
        self.add_input('edge_size', units="cm")
        self.add_input('height_size', units="cm")

        self.add_output('h_size', units="N*m*s")
        self.add_output('Mrw', units="kg") #shared variable
        self.add_output('Prw', units="W") #shared variable

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        M_approx_total = inputs['M_total']
        Rwr = inputs['Rwr']
        Rwav = inputs['Rwav']
        Op = inputs['Op']
        t_slew = inputs['t_slew']
        MaxSA = inputs['MaxSA']
        edge_size = inputs['edge_size']
        height_size = inputs['height_size']

        Icubesat = 1/12 *  M_approx_total * ((edge_size*0.01)**2 + (height_size*0.01)**2)
        T_slew = 4 * (MaxSA * math.pi / 180) * Icubesat / (t_slew ** 2)
        h_slew = T_slew * t_slew
        h_size = 3*h_slew
        Mrw = h_size / (Rwav * Rwr**2)
        #Prw = Mrw * h_size

        alpha = 0.01
        Torque = Icubesat * alpha
        Pmech = Torque * Rwav
        MotorEfficiency = 0.8
        Prw = Pmech / MotorEfficiency

        outputs['h_size'] = h_size
        outputs['Mrw'] = Mrw
        outputs['Prw'] = Prw

class MomentumWheelNew(om.ExplicitComponent):
    """Sizing of momentum wheel"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        self.add_input('Op', units="min")
        self.add_input('T_total', val=varval('Constants/Mission/T_total'), units="N*m")
        self.add_input('Mwr', val=varval('ADCS/Momentum_wheel/Mwr'), units="m")
        self.add_input('Mwav', val=varval('ADCS/Momentum_wheel/Mwav'), units="rad/s")
        self.add_input('Maad', val=varval('ADCS/Momentum_wheel/Maad'), units="deg")
        self.add_input('M_total', units="kg") #shared variable
        self.add_input('edge_size', units="cm")
        self.add_input('height_size', units="cm")

        self.add_output('Mram', units="N*m*s")
        self.add_output('Mmw', units="kg") #shared variable
        self.add_output('Pmw', units="W") #shared variable

        self.declare_partials(of='*', wrt='*', method='fd')

    def compute(self, inputs, outputs):

        Op = inputs['Op']
        T_total = inputs['T_total']
        Mwr = inputs['Mwr']
        Mwav = inputs['Mwav']
        Maad = inputs['Maad']
        M_approx_total = inputs['M_total']
        edge_size = inputs['edge_size']
        height_size = inputs['height_size']

        Mram = T_total * Op * 60 / (4 * Maad * (math.pi / 180))
        Mmw = Mram / (Mwav * Mwr ** 2)

        Icubesat = 1 / 12 * M_approx_total * ((edge_size * 0.01) ** 2 + (height_size * 0.01) ** 2)
        alpha = 0.01
        Torque = Icubesat * alpha
        Pmech = Torque * Mwav
        MotorEfficiency = 0.8
        Pmw = Pmech / MotorEfficiency

        outputs['Mram'] = Mram
        outputs['Mmw'] = Mmw
        outputs['Pmw'] = Pmw


"""
4. Power Subsystem: Solar Panel, Secondary Battery, and Primary Battery
"""

class SolarPanel(om.ExplicitComponent):
    """Solar panel sizing"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        self.add_input('P_total', units="W") #shared variable
        self.add_input('f_sunlight', val=varval('Power/Solar_array/f_sunlight'))
        self.add_input('PDPY', val=varval('Power/Solar_array/PDPY'))
        self.add_input('SF', val=varval('Constants/Earth/SF'), units="W/m**2")
        self.add_input('Efficiency', val=varval('Power/Solar_array/Efficiency'))
        self.add_input('Wcsia', val=varval('Constants/Mission/Wcsia'), units="deg")
        self.add_input('Area_density', val=varval('Power/Solar_array/Area_density'), units="kg/m**2")

        self.add_output('P_generated', units="W")
        self.add_output('A_solarpanel', units="m**2")
        self.add_output('m_solarpanel', units="kg") #shared variable

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        P_total = inputs['P_total']
        f_sunlight = inputs['f_sunlight']
        PDPY = inputs['PDPY']
        SF = inputs['SF']
        Efficiency = inputs['Efficiency']
        Wcsia = inputs['Wcsia']
        Area_density = inputs['Area_density']

        P_average = P_total / f_sunlight
        P_generated = P_average / (1-PDPY)
        Wcsia_deg = np.radians(Wcsia)
        P_area = SF * Efficiency * np.cos(Wcsia_deg)
        A_solarpanel = P_generated / P_area
        m_solarpanel = Area_density * A_solarpanel

        outputs['P_generated'] = P_generated
        outputs['A_solarpanel'] = A_solarpanel
        outputs['m_solarpanel'] = m_solarpanel

class SecondaryBatteryNew(om.ExplicitComponent):
    """Secondary battery sizing"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        self.add_input('MaxET', val=varval('Constants/Mission/MaxET'), units="min")
        self.add_input('P_total', units="W") #shared variable
        self.add_input('DoD', val=varval('Power/Secondary_battery/DoD'))
        self.add_input('Ed', val=varval('Power/Secondary_battery/Ed'))

        self.add_output('E_battery')
        self.add_output('Msb', units="kg") #shared variable

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        MaxET = inputs['MaxET']
        P_total = inputs['P_total']
        DoD = inputs['DoD']
        Ed = inputs['Ed']

        Margin = 0.5
        E_eclipse = (1+Margin) * P_total * MaxET*60/3600
        E_battery = E_eclipse/DoD
        Msb = E_battery / Ed

        outputs['E_battery'] = E_battery
        outputs['Msb'] = Msb

class PrimaryBatteryNew(om.ExplicitComponent):
    """Primary battery sizing"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        self.add_input('MD', val=varval('Constants/Mission/MD'))
        self.add_input('P_total', units="W")
        self.add_input('e_pb', val=varval('Power/Primary_battery/e_pb'))

        self.add_output('E_battery')
        self.add_output('Mpb', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        MD = inputs['MD']
        P_total = inputs['P_total']
        e_pb = inputs['e_pb']

        Margin = 0.2
        E_battery = (1+Margin) * P_total * MD*24
        Mpb = E_battery / e_pb

        outputs['E_battery'] = E_battery
        outputs['Mpb'] = Mpb


"""
5. Payload Subsystem: Remote Sensing
"""

class PayloadRemoteSensing(om.ExplicitComponent):
    """Sizing of payload for remote sensing"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        self.add_input('x_optical', val=varval('Payload/x_optical'), units="m", desc="Pixel size")
        self.add_input('Alt', units="km", desc="Orbit altitude")
        self.add_input('GSD_optical', val=varval('Payload/GSD_optical'), units="m", desc="Ground Sampling Distance")
        self.add_input('lambda_optical', val=varval('Payload/lambda_optical'), units="m", desc="Wave length")
        self.add_input('Q_optical', val=varval('Payload/Q_optical'), desc="Image quality")

        self.add_output('f_optical', units="mm", desc="Focal length")
        self.add_output('D_optical', units="mm", desc="Aperture diameter")
        self.add_output('m_payload', units="kg") #shared variable
        self.add_output('P_payload', units="W") #shared variable
        self.add_output('l_payload', units="m", desc="length of payload")
        self.add_output('w_payload', units="m", desc="width of payload")
        self.add_output('h_payload', units="m", desc="height of payload")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        x_optical = inputs['x_optical']
        Alt = inputs['Alt']
        GSD_optical = inputs['GSD_optical']
        lambda_optical = inputs['lambda_optical']
        Q_optical = inputs['Q_optical']

        f_optical = x_optical * Alt*1000 / GSD_optical
        D_optical = (lambda_optical * f_optical) / (Q_optical * x_optical)

        R = f_optical*1000/70
        K = 1
        m_payload = K * R**3 * 0.277
        P_payload = K * R**3 * 1.3

        l_payload = R * 0.096
        w_payload = R * 0.090
        h_payload = R * 0.058

        outputs['f_optical'] = f_optical
        outputs['D_optical'] = D_optical
        outputs['m_payload'] = m_payload
        outputs['P_payload'] = P_payload
        outputs['l_payload'] = l_payload
        outputs['w_payload'] = w_payload
        outputs['h_payload'] = h_payload


"""
6. Communication Subsystem: Radio Transceiver 
"""

class CommunicationRadioTransceiver(om.ExplicitComponent):
    """Sizing model of radio transceiver as communication component"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        # New inputs
        self.add_input('f_down', val=varval('Communication/Downlink/f_down'), units="Hz", desc="Carrier frequency")
        self.add_input('P_t_down', val=varval('Communication/Downlink/P_t_down'), units="W", desc="Transmitter power")
        self.add_input('L_l_down', val=varval('Communication/Downlink/L_l_down'), desc="Transmitter line loss")
        self.add_input('theta_t_down', val=varval('Communication/Downlink/theta_t_down'), units="deg", desc="Transmit antenna beamwidth")
        self.add_input('e_t_down', val=varval('Communication/Downlink/e_t_down'), units="deg", desc="Transmit antenna pointing offset")
        self.add_input('S_down', val=varval('Communication/Downlink/S_down'), units="km", desc="Propagation path length")
        self.add_input('L_a_down', val=varval('Communication/Downlink/L_a_down'), desc="Propagation and polarization loss") # might be an output
        self.add_input('eff_down', val=varval('Communication/Downlink/eff_down'), desc="Antenna efficiency")
        self.add_input('D_r_down', val=varval('Communication/Downlink/D_r_down'), units="m", desc="Receive antenna diameter")
        self.add_input('e_r_down', val=varval('Communication/Downlink/e_r_down'), units="deg", desc="Receive antenna pointing error")
        self.add_input('T_s_down', val=varval('Communication/Downlink/T_s_down'), units="K", desc="System noise temperature") # might be an output
        self.add_input('R_down', val=varval('Communication/Downlink/R_down'), units="s**(-1)", desc="Data rate")
        self.add_input('BER_down', val=varval('Communication/Downlink/BER_down'), desc="Bit Error Rate")
        self.add_input('L_imp_down', val=varval('Communication/Downlink/L_imp_down'), desc="Implementation loss")
        self.add_input('Eb_No_req_down', val=varval('Communication/Downlink/Eb_No_req_down'), desc="Required system-to-noise ratio") # might be an output

        # Outputs
        self.add_output('M_comm_down', units="kg", desc="Mass communication subsystem") # shared var
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
        self.add_output('Br_down', desc="data downloaded")

        # Derivatives declaration
        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        # New inputs
        f = inputs['f_down']
        P_t = inputs['P_t_down']
        L_l = inputs['L_l_down']
        theta_t = inputs['theta_t_down']
        e_t = inputs['e_t_down']
        S = inputs['S_down']
        L_a = inputs['L_a_down']
        eff = inputs['eff_down']
        D_r = inputs['D_r_down']
        e_r = inputs['e_r_down']
        T_s = inputs['T_s_down']
        R = inputs['R_down']
        BER = inputs['BER_down']
        L_imp = inputs['L_imp_down']
        Eb_No_req = inputs['Eb_No_req_down']

        # Sizing model
        P_t_dB = 10 * np.log10(P_t) #transmitter power conversion to dB
        G_pt = 44.3 - 10 * np.log10(theta_t**2) #peak transmit antenna gain (eq. 13-20)
        D_t = 21 / (f * theta_t) #transmit antenna diameter (eq. 13-19)
        L_pt = -12 * (e_t/theta_t)**2 #transmit antenna pointing loss (eq. 13-21)
        G_t = G_pt + L_pt #transmit antenna gain
        EIRP = P_t_dB + L_l + G_t #equivalence isotropic radiated power
        L_s = 20 * np.log10(3e8) - 20 * np.log10(4 * np.pi) - 20 * np.log10(S * 1000) - 20 * np.log10(f) - 180.0  # Space loss
        G_rp = -159.59 + 20 * np.log10(D_r) + 20 * np.log10(f) + 10 * np.log10(eff) + 180.0 #peak receive antenna gain (eq. 13-18)
        theta_r = 21 / (f * D_r) #receive antenna beamwidth (eq. 13-19)
        L_pr = -12 * (e_r/theta_r)**2 #receive antenna pointing loss (eq. 13-21)
        G_r = G_rp + L_pr #receive antenna gain
        Eb_No = P_t_dB + L_l + G_t + L_pr + L_s + L_a + G_r + 228.6 - 10 * np.log10(T_s) - 10 * np.log10(R)
        C_No = Eb_No + 10 * np.log10(R)
        Margin = Eb_No - Eb_No_req + L_imp

        #Statistical approach, with the reference of EnduroSat p.259 SOTA of Small Spacecraft NASA
        R = f/2.2
        M_comm = R**3 * 0.195
        P_comm = R**3 * 1.25

        Br_down = (3 * 10 ** 8 * G_r * L_l / (16 * math.pi ** 2 * f * T_s * Eb_No)) * (eff * P_comm * G_r / S ** 2)

        outputs['M_comm_down'] = M_comm
        outputs['P_comm_down'] = P_comm
        outputs['Br_down'] = Br_down
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


"""
7. On-Board Computer and Data Handling Subsystem (OBCS): ARM Cortex-M
"""

class OBCSARMCortexM(om.ExplicitComponent):
    """Sizing of On-Board Computer and Data Handling Subsystem: ARM Cortex-M"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        self.add_input('processor_speed', val=varval('OBCS/processor_speed'), units="GHz")
        self.add_input('num_cores', val=varval('OBCS/num_cores'))
        self.add_input('memory_size', val=varval('OBCS/memory_size'))
        self.add_input('num_data_channels', val=varval('OBCS/num_data_channels'))

        self.add_output('P_OBCS', units="W")
        self.add_output('m_OBCS', units="kg")
        self.add_output('data_rate')
        self.add_output('memory_usage')

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        processor_speed = inputs['processor_speed']
        num_cores = inputs['num_cores']
        memory_size = inputs['memory_size']
        num_data_channels = inputs['num_data_channels']

        # ARM Cortex-M characteristics (typical for embedded systems)
        base_power_per_core = 0.05  # Base power consumption per core (W) for ARM Cortex-M
        base_power_per_memory = 0.01  # Power per GB of memory (W)
        base_power_per_channel = 0.2  # Power consumption per data channel (W)
        max_data_rate_per_core = 100
        memory_usage_per_channel = 0.05
        power_per_core = base_power_per_core

        P_OBCS = (num_cores * power_per_core + memory_size * base_power_per_memory +
                  num_data_channels * base_power_per_channel)
        data_rate = processor_speed * num_cores * max_data_rate_per_core
        memory_usage = memory_size + num_data_channels * memory_usage_per_channel
        m_OBCS = 0.13 #fixed-value from https://www.endurosat.com/products/onboard-computer/

        outputs['P_OBCS'] = P_OBCS
        outputs['m_OBCS'] = m_OBCS
        outputs['data_rate'] = data_rate
        outputs['memory_usage'] = memory_usage


"""
8. Miscellaneous Components: Components with no analytical equations, properties taken directly from COTS
"""

class MiscellaneousComponents(om.ExplicitComponent):
    """Constant values of mass and power for other components, such as sensor, antenna, cable, etc"""

    def setup(self):
        def varval(name):
            tree = ET.parse('C:/Users/marco.wijaya/PycharmProjects/CADREforSAMP/Sizing/input_file_CubeSat.xml')
            element = tree.getroot().find(name)

            if element is not None:
                value = element.text.strip()
            else:
                value = None
            return value

        self.add_input('m_antenna', val=varval('Miscellaneous/m_antenna'), units="kg") #https://www.endurosat.com/products/s-band-antenna-commercial/
        self.add_input('m_sensor', val=varval('Miscellaneous/m_sensor'), units="kg") #sun sensor (solar MEMS Technologies) from NASA's SOTA
        self.add_input('m_cable', val=varval('Miscellaneous/m_cable'), units="kg")
        self.add_input('P_antenna', val=varval('Miscellaneous/P_antenna'), units="W")
        self.add_input('P_sensor', val=varval('Miscellaneous/P_sensor'), units="W") #sun sensor (solar MEMS Technologies) from NASA's SOTA

        self.add_output('m_misc', units="kg")
        self.add_output('P_misc', units="W")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        m_antenna = inputs['m_antenna']
        m_sensor = inputs['m_sensor']
        m_cable = inputs['m_cable']
        P_antenna = inputs['P_antenna']
        P_sensor = inputs['P_sensor']

        m_misc = m_antenna + m_sensor + m_cable
        P_misc = P_antenna + P_sensor

        outputs['m_misc'] = m_misc
        outputs['P_misc'] = P_misc


"""
Mass Aggregation Class
"""

class MassAggregation(om.ExplicitComponent):
    """Mass aggregation from all components of CubeSat"""

    def setup(self):

        self.add_input('M_thermal', units="kg")
        self.add_input('M_structure', units="kg")
        self.add_input('Mrw', units="kg")
        self.add_input('Mmw', units="kg")
        self.add_input('m_solarpanel', units="kg")
        self.add_input('Msb', units="kg")
        self.add_input('m_payload', units="kg")
        self.add_input('M_comm_down', units="kg")
        self.add_input('m_OBCS', units="kg")
        self.add_input('m_misc', units="kg")

        self.add_output('M_total', units="kg")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        M_thermal = inputs['M_thermal']
        M_structure = inputs['M_structure']
        Mrw = inputs['Mrw']
        Mmw = inputs['Mmw']
        m_solarpanel = inputs['m_solarpanel']
        Msb = inputs['Msb']
        m_payload = inputs['m_payload']
        M_comm_down = inputs['M_comm_down']
        m_OBCS = inputs['m_OBCS']
        m_misc = inputs['m_misc']

        M_total = M_thermal + M_structure + Mrw + Mmw + m_solarpanel + Msb + m_payload + M_comm_down + m_OBCS + m_misc

        outputs['M_total'] = M_total


"""
Power Aggregation Class
"""

class PowerAggregation(om.ExplicitComponent):
    """Power aggregation from all components of CubeSat who consume power"""

    def setup(self):
        self.add_input('Prw', units="W")
        self.add_input('Pmw', units="W")
        self.add_input('P_payload', units="W")
        self.add_input('P_comm_down', units="W")
        self.add_input('P_OBCS', units="W")
        self.add_input('P_misc', units="W")

        self.add_output('P_total', units="W")

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):

        Prw = inputs['Prw']
        Pmw = inputs['Pmw']
        P_payload = inputs['P_payload']
        P_comm_down = inputs['P_comm_down']
        P_OBCS = inputs['P_OBCS']
        P_misc = inputs['P_misc']

        P_total = Prw + Pmw + P_payload + P_comm_down + P_OBCS + P_misc

        outputs['P_total'] = P_total


"""
MDO Wrapper of All Subsystems
"""

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
    CubeSat.Payload.add_subsystem('RemoteSensing',PayloadRemoteSensing(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_subsystem('Communication', om.Group(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.Communication.add_subsystem('RadioTransceiver', CommunicationRadioTransceiver(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_subsystem('Aggregation', om.Group(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.Aggregation.add_subsystem('MassAggregation', MassAggregation(), promotes_inputs=['*'], promotes_outputs=['*'])
    CubeSat.Aggregation.add_subsystem('PowerAggregation', PowerAggregation(), promotes_inputs=['*'], promotes_outputs=['*'])

    CubeSat.add_design_var('Area_density', lower=1.0, upper=3.0)
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
