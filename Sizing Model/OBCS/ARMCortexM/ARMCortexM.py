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
class OBCSARMCortexM(om.ExplicitComponent):
    """Sizing of On-Board Computer and Data Handling Subsystem: ARM Cortex-M"""

    processor_speed: float
    num_cores: float
    memory_size: float
    num_data_channels: float

    # P_OBCS: float
    # m_OBCS: float
    # data_rate: float
    # memory_usage: float

    def __init__(self, processor_speed, num_cores, memory_size, num_data_channels):
        super().__init__()

        self.processor_speed = processor_speed
        self.num_cores = num_cores
        self.memory_size = memory_size
        self.num_data_channels = num_data_channels
        # self.P_OBCS = P_OBCS
        # self.m_OBCS = m_OBCS
        # self.data_rate = data_rate
        # self.memory_usage = memory_usage

    def setup(self):
        self.add_input('processor_speed', val=self.processor_speed, units="GHz")
        self.add_input('num_cores', val=self.num_cores)
        self.add_input('memory_size', val=self.memory_size)
        self.add_input('num_data_channels', val=self.num_data_channels)

        self.add_output('P_OBCS', units="W")
        self.add_output('m_OBCS', units="kg")
        self.add_output('data_rate')
        self.add_output('memory_usage')

        self.declare_partials(of=['*'], wrt=['*'], method='fd')

    def compute(self, inputs, outputs):
        self.processor_speed = inputs['processor_speed']
        self.num_cores = inputs['num_cores']
        self.memory_size = inputs['memory_size']
        self.num_data_channels = inputs['num_data_channels']

        # ARM Cortex-M characteristics (typical for embedded systems)
        base_power_per_core = 0.05  # Base power consumption per core (W) for ARM Cortex-M
        base_power_per_memory = 0.01  # Power per GB of memory (W)
        base_power_per_channel = 0.2  # Power consumption per data channel (W)
        max_data_rate_per_core = 100
        memory_usage_per_channel = 0.05
        power_per_core = base_power_per_core

        self.P_OBCS = (self.num_cores * power_per_core + self.memory_size * base_power_per_memory +
                       self.num_data_channels * base_power_per_channel)
        self.data_rate = self.processor_speed * self.num_cores * max_data_rate_per_core
        self.memory_usage = self.memory_size + self.num_data_channels * memory_usage_per_channel
        self.m_OBCS = 0.13  # fixed-value from https://www.endurosat.com/products/onboard-computer/

        outputs['P_OBCS'] = self.P_OBCS
        outputs['m_OBCS'] = self.m_OBCS
        outputs['data_rate'] = self.data_rate
        outputs['memory_usage'] = self.memory_usage