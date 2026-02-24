# Introduction to GEM5


## What is GEM5 

### **History of gem5**

- **M5 Simulator:**
  - Developed at the University of Michigan by Steve Reinhardt and students.
  - Created for system simulation with event-driven architecture.
- **GEMS (Multiac Gyms):**
  - Developed at the University of Wisconsin by Mark Hill and David Wood’s students.
  - Focused on detailed memory system simulations.
- **Merger into gem5 (2011):**
  - M5 and GEMS combined to create a unified modular platform.
  - gem5 became a simulator for computer system architecture research.

### **Goals of gem5**
- Simulate the entire computer system stack, from application to hardware.
- Support agile hardware development for fast design space exploration.
- Enable research at different system levels, including kernel, runtime, and hardware co-design.
- Facilitate full-stack simulation for emerging applications (ML, AR/VR, etc.).
- Encourage community contributions to improve gem5.

### **gem5 Community**
- Hundreds of contributors, including academic researchers and industry professionals (Google, AMD, Arm, Samsung, Micron, etc.).
- Supports research, industry R&D, and classroom use.
- Encourages open-source collaboration and contributions.
- [gem5 events](https://www.gem5.org/events/)

### **gem5’s Software Architecture**
- **Combination of C++ and Python:**
  - C++ for core simulation models.
  - Python for configuration and control.
- **Simulation Objects (SimObjects):**
  - Models for CPU, caches, memory, and devices.
  - Highly parameterized for flexible system design.
- **Python Wrappers:**
  - Used to create and control simulation instances.
  - Standard Library provides pre-configured components.
- **Event-Driven Simulation:**
  - Uses an event queue with scheduled execution.
    - Example: Fetching the instruction at the next PC:
      - Event 1. Fetch instruction
      - Event 2. Request instruction from cache
      - Event 3. Cache response: miss -> fetch from memory
      - Event 4. Send the request to DRAM
      - Event 5. Read the instruction from DRAM
      - Event 6. Get the instruction from the DRAM
      - Event 7. Cache receives the instruction 
      - Event 8. CPU fetches the instruction from the cache
      - Event 9. CPU decodes the instruction
      - Event 10. CPU executes the instruction
      - Event 11. CPU next instruction fetch begins
    - Note: events are scheduled e.g. we need to wait for the DRAM latency to get the instruction from the memory, etc. 
  
  - Models time with "ticks" (default: 1 picosecond per tick).

### **Key Abstractions in gem5**

1. **Memory System (Ports and Requests):**
   - Unidirectional ports connect components.
   - Allows flexible configuration of memory hierarchy.
2. **CPU and ISA Separation:**
   - ISA-independent CPU models.
   - Allows different instruction sets to work with the same CPU model.
                |


<img src=./img/gem5_intro.png width="200">


## GEM5 Standard Library

The gem5 standard library is designed to provide users with commonly used components, features, and functionality to improve their productivity.

### **The gem5 stdlib components**

- The central part of the gem5 stdlib allows users to build complex systems from simple components that connect using standardized APIs.
- At the core of the components package is the concept of a board, which exposes standardized interfaces for adding other hardware and handling communication between them.

    - Three elementary subcomponents:
        - **Processor**
        - **Cache Hierarchies**
        - **Memory System**
  
## GEM5 Hello World

Let's start with a simple example of a gem5 simulation. The following code snippet shows a simple gem5 simulation that creates a system with a CPU, a memory system without caches. 

The scripts begin by importing the necessary components from the gem5 library.

```python
from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.resources.resource import CustomResource
from gem5.simulate.simulator import Simulator
from gem5.isas import ISA

```

Next, we create a cache hierarchy and a memory system. The cache hierarchy is set to `NoCache`, which means that there are no caches in the system. The memory system is set to `SingleChannelDDR3_1600`, which is a single-channel DDR3 memory system running at 1600 MHz.

```python
cache_hierarchy = NoCache()
memory = SingleChannelDDR3_1600()
```
Next, we create a processor and set its type to `CPUTypes.ATOMIC`. We will talk more about the different processor types in the next exercsises.

```python
processor = SimpleProcessor(cpu_type=CPUTypes.ATOMIC, num_cores=1, isa=ISA.RISCV)
```

The last step in creating the system is to create a board and add the components to it. The board is a container for the components and provides a standardized interface for adding and connecting components.

```python
board = SimpleBoard(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)
```

### Simulating the system 

So far, we have described the system components and how they are connected. The next step is to simulate the system. When simulating the system two things are important:
- The workload that the system will run.
- Simulation setup

The workload for the simulation is set using a binary resource:

```python
# Set the workload.
binary = CustomResource("path/to/hello_world")
board.set_se_binary_workload(binary)
```

Finally, the simulator is set up and the simulation is run:

```python
# Setup the Simulator and run the simulation.
simulator = Simulator(board=board)
simulator.run()
```

#### Types of simulation:

**System Emulation (SE) mode:**
In System Emulation mode, gem5 runs user-level applications on top of a lightweight emulated environment without booting a full operating system. The simulator provides simplified OS services (e.g., system calls) and focuses on fast functional and performance evaluation of application and microarchitectural behavior, making SE mode ideal for early-stage architectural exploration and benchmarking.

**Full System (FS) mode:**
In Full System mode, gem5 simulates an entire computer system, including the boot process, operating system, device drivers, and peripherals. This mode enables detailed evaluation of interactions between hardware and the OS, making it suitable for studying system software, kernel behavior, I/O subsystems, and full-stack performance, albeit with higher simulation complexity and longer runtime.

## How to compile the code

Our workload is a simple "Hello world" program that we will compile for the RISC-V architecture. To do so, we compile the code using the RISC-V toolchain, which is available in the apptainer container, that is used for the exercises. The bash script `compile` in the `workload` directory compiles the code for the RISC-V architecture and creates a binary that can be used as a workload for our gem5 simulation.


## How to install GEM5

Please refer to the official [gem5 installation guide](https://www.gem5.org/documentation/learning_gem5/part1/building/).

## How to run GEM5

After we have build the gem5, we can run the simulation using the following command:

```bash
$GEM5/build/ALL/gem5.opt --outdir=hello_stats hello_world.py 
```

where `outdir` is the directory where the simulation statistics will be stored and `hello_world.py` is the script that we have created in the previous section.

## Literature

- [gem5.org](https://www.gem5.org/)
- [gem5 2024 Bootcamp - Introduction](https://gem5bootcamp.github.io/2024/#01-Introduction/00-introduction-to-bootcamp)
- [gem5 Standard Libraries](https://www.gem5.org/documentation/gem5-stdlib/overview)
- [gem5 Hello world](https://www.gem5.org/documentation/gem5-stdlib/hello-world-tutorial)
