
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.memory.multi_channel import DualChannelDDR4_2400
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_switchable_processor import (
    SimpleSwitchableProcessor
)
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import obtain_resource
from gem5.simulate.exit_event import ExitEvent
from gem5.simulate.simulator import Simulator
from gem5.resources.resource import CustomResource
from gem5.components.memory.single_channel import SingleChannelDDR3_1600

from three_level import PrivateL1PrivateL2SharedL3CacheHierarchy

import m5




cache_hiearchy = PrivateL1PrivateL2SharedL3CacheHierarchy(
    l1d_size="32KiB",
    l1d_assoc=8,
    l1i_size="32KiB",
    l1i_assoc=8,
    l2_size="256KiB",
    l2_assoc=8,
    l3_size="2MiB",
    l3_assoc=32,
)

processor = SimpleProcessor(
        cpu_type=CPUTypes.MINOR,
        num_cores=4  ,
        isa=ISA.RISCV,
    )


memory = SingleChannelDDR3_1600(size="4GiB")




#add board 
board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hiearchy
)

binary = CustomResource("./workload/imbalanced/work_imbalanced.bin")
#binary = CustomResource("./workload/variables/mat_vec_mult.bin")
#binary = CustomResource("./workload/vec_add/vec_add.bin")
board.set_se_binary_workload(binary)

simulator = Simulator(board=board)
simulator.run()
