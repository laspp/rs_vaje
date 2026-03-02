from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.private_l1_cache_hierarchy import PrivateL1CacheHierarchy
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA
from gem5.resources.resource import obtain_resource
from gem5.simulate.simulator import Simulator
from gem5.resources.resource import CustomResource
from cpuInORD_model import RiscV_InOrder_CPU



cache_hierarchy = PrivateL1CacheHierarchy(l1d_size="64KiB", l1i_size="64KiB")


memory = SingleChannelDDR3_1600("7GiB")

# By default, use Atomic CPU
#cpu_type = CPUTypes.ATOMIC


# Uncomment for Timing CPU
cpu_type = CPUTypes.TIMING

#processor = SimpleProcessor(cpu_type=cpu_type, num_cores=1, isa=ISA.RISCV)

#Source code for simple processor gem5/src/python/gem5/components/processors/simple_processor.py
processor = RiscV_InOrder_CPU()


board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy
)

# Resources can be found at
    # https://resources.gem5.org/
# x86-matrix-multiply is obtained from
    # https://resources.gem5.org/resources/x86-matrix-multiply-run?version=1.0.0



# Set the workload.
binary = CustomResource("./workload/mat_mult/matrix_mult.bin")
board.set_se_binary_workload(binary)

simulator = Simulator(board=board)
simulator.run()