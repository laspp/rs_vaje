# Ruby Cache Coherence Modeling

- The Ruby memory system in GEM5 provides a more detailed and accurate representation of the memory subsystem. It models:
    - cache hierarchies 
    - coherence protocols
    - interconnection networks
    - memory and DMA controllers
    - various sequencers that manage the flow of data between these components.

- The Ruby memory system is more complex than the classic one but provides a more accurate representation of modern memory subsystems. It is beneficial for modeling complex cache hierarchies, coherence protocols, and interconnection networks in multicore systems.

- Modeling cache coherence protocols with Ruby involves describing it using SLICC (Specification Language for Implementing Cache Coherence). SLICC is a domain-specific language that allows you to define the coherence protocol, cache hierarchy, and other memory system components in a high-level, abstract manner. We will rely on the existing coherence protocols implemented in GEM5 for the laboratory exercises, but you can also create your custom protocols using SLICC. The following cache coherence protocols are available in GEM5:
    - MSI_Two_Level
    - MESI_Two_Level
    - MOESI_CMP_directory
    - MOESI_CMP_token
    - MOESI_hammer 


<!---

---
**NOTE**

In order to use CC protocols, you need to recompile GEM5 with options that enable them. Steps:

1. Go to the GEM5 directory.
2. In folder `build_opts/`, create a file named `X86_ALL_RUBY` with the following content:
```
RUBY=y
USE_MULTIPLE_PROTOCOLS=y
PROTOCOL="MULTIPLE"
RUBY_PROTOCOL_MOESI_AMD_Base=y
RUBY_PROTOCOL_MESI_Two_Level=y
RUBY_PROTOCOL_MESI_Three_Level=y
RUBY_PROTOCOL_MESI_Three_Level_HTM=y
RUBY_PROTOCOL_MI_example=y
RUBY_PROTOCOL_MOESI_CMP_directory=y
RUBY_PROTOCOL_MOESI_CMP_token=y
RUBY_PROTOCOL_MOESI_hammer=y
RUBY_PROTOCOL_Garnet_standalone=y
RUBY_PROTOCOL_CHI=y
RUBY_PROTOCOL_MSI=y
BUILD_ISA=y
USE_X86_ISA=y
```
4. Copy the files in folder `utils/` to `/d/hpc/home/ratkop/gem5_workspace/gem5/src/mem/ruby/protocol/:

3. Run the following command to recompile GEM5:
```
scons build/X86_ALL_RUBY/gem5.opt -j9
```
4. For next simulations, use the `X86_ALL_RUBY/gem5.opt` binary file.
---
--->

## Ruby memory system components

1. **Caches**: The caches in the Ruby memory system are built upon RubyCache objects.

   - Each protocol features its own cache implementation that extends the RubyCache class.
   - The cache consists of two main components:
     - The cache itself
     - The interface to the network:
       - The cache includes queues for outgoing requests and responses.
       - The cache also has queues for incoming requests and responses.
   - The mandatory queue serves as the interface between the Sequencer and the SLICC-generated cache coherence files.

   - **L1 Queues**:
     - The L1 cache has two primary queues:
       - The `*ToL1` queue is used to send requests from the L2 cache to the L1 cache.
       - The `*FromL1` queue is used to send responses from the L1 cache back to the L2 cache.
       - **Unblock Messages**: The block in the L1 cache remains blocked until an acknowledgment is received from the sharers in the directory. 
   
   - **L2 Queues**:
     - There are three different queues:
       - Requests from and to the L1 cache.
       - Requests from and to the directory controllers.
       - Responses from and to the L2 cache.
        

2. **Sequencers**: Sequencers feed the memory object with load, store, and atomic memory requests from the processor. They also send the responses back to the processor.

3. **Directory Controllers**: Directory controllers are responsible for managing the directory of a memory region. They play a key role in protocols that utilize a directory-based coherence mechanism. 

    - **Queues**:
        - The directory controller maintains queues for incoming requests.
        - The directory controller has queues for outgoing responses.
        - The directory controller has queues for incoming responses.
        - The directory controller includes queues for requests to the memory controller.


4. **DMA Controllers**: The DMA controllers manage the flow of data between memory and I/O devices.

5. **Interconnection Network**: The interconnection network links the different components of the memory hierarchy, including the cache, memory, and DMA controllers.

## Modeling multicore systems with Ruby

1. The cache hiearchy object is built upon:
    - AbstractRubyCacheHierarchy class
    - AbstractTwoLevelCacheHierarchy class

2. Initialize the parameters of cache hiearchy and create the AbstractRubyCacheHierarchy and AbstractTwoLevelCacheHierarchy objects.
    ```python
    def __init__(
        self,
        l1i_size: str,
        l1i_assoc: str,
        l1d_size: str,
        l1d_assoc: str,
        l2_size: str,
        l2_assoc: str,
        num_l2_banks: int,
    ):
        AbstractRubyCacheHierarchy.__init__(self=self)
        AbstractTwoLevelCacheHierarchy.__init__(
            self,
            l1i_size=l1i_size,
            l1i_assoc=l1i_assoc,
            l1d_size=l1d_size,
            l1d_assoc=l1d_assoc,
            l2_size=l2_size,
            l2_assoc=l2_assoc,
        )

        self._num_l2_banks = num_l2_banks
    ```

3. Get coherence protocol 
    ```python
    @overrides(AbstractCacheHierarchy)
    def get_coherence_protocol(self):
        return CoherenceProtocol.MESI_TWO_LEVEL
    ```

4. Creating the Ruby system

    a. Create a RubySystem object, which encapsulates the Ruby memory system. The objects contains all of the various parts of the system we are simulating. Performs allocation, deallocation, and setup of all the major components of the system 

    b. Set the number of virtual networks in the Ruby memory system. The number of virtual networks is the number of separate networks that can be used to send packets. The number of virtual network is determined by the coherence protocol being used. For example, the [MESI_Two_Level protocol](https://github.com/gem5/gem5/blob/stable/src/mem/ruby/protocol/MESI_Two_Level-L1cache.sm) requires three virtual networks.

    c. Create a network object that will be used to connect the various components of the Ruby memory system. The network object is responsible for routing packets between the various components of the system. The [SimplePt2Pt](https://www.gem5.org/documentation/general_docs/stdlib_api/gem5.components.cachehierarchies.ruby.topologies.simple_pt2pt.html) network is a simple point-to-point network that connects  all of the controllers to routers and connec the routers together in a point-to-point network.
    

    ```python
    def incorporate_cache(self, board: AbstractBoard) -> None:
        super().incorporate_cache(board)
        cache_line_size = board.get_cache_line_size()
        # a) Create a RubySystem object, which encapsulates the Ruby memory system
        self.ruby_system = RubySystem()

        # b) Set the number of virtual networks in the Ruby memory system
        self.ruby_system.number_of_virtual_networks = 3

        # c) Create a network object that will be used to connect the various components of the Ruby memory system
        self.ruby_system.network = SimplePt2Pt(self.ruby_system)
        self.ruby_system.network.number_of_virtual_networks = 3
    ``` 


5.  Creating the L1 cache and sequencer for each core

    a) First, create the L1 cache for each core. The L1 cache is built on top of a RubyCache object, representing the Ruby memory system cache. Additionally, it contains MessageBuffers for transmitting requests and responses.

    b) Next, we need to create the sequencer for the L1 caches. The sequencer handles memory requests—such as load, store, and atomic operations—from the processor. Additionally, it sends responses back to the processor.

    c) After that, connect the sequencer to the CPU. This is done by linking the input ports of the sequencer to the instruction cache (icache) and data cache (dcache) of the core.

    d) The remaining code connects the Memory Management Unit (MMU), interrupt controller, and I/O bus to the sequencer.

    ```python
        self._l1_controllers = []
        for i, core in enumerate(board.get_processor().get_cores()):
            # a) For each core, create an L1 cache and connect it to the core. Also create sequencer for each L1 cache.
            cache = L1Cache(
                self._l1i_size,
                self._l1i_assoc,
                self._l1d_size,
                self._l1d_assoc,
                self.ruby_system.network,
                core,
                self._num_l2_banks,
                cache_line_size,
                board.processor.get_isa(),
                board.get_clock_domain(),
            )

            # b) Create sequencer for L1 cache
            cache.sequencer = RubySequencer(
                version=i,
                dcache=cache.L1Dcache,
                clk_domain=cache.clk_domain,
                ruby_system=self.ruby_system,
            )

            # register the cache with the ruby system
            cache.ruby_system = self.ruby_system

            # c) connect the sequencer to the CPU 
            core.connect_icache(cache.sequencer.in_ports)
            core.connect_dcache(cache.sequencer.in_ports)
    ```

6. Create the L2 caches

    a) Similarly,we create the L2 caches. Unlike the previous exercise, the L2 caches are not private to each core; instead, they are shared among all the cores in the system. This cache will feature distinct queues for requests and responses, to manage incoming requests from the L1 cache and the directory controllers.

    b) Next, we register the L2 cache with the Ruby memory system.


    ```python
    # Create the L2 cache controllers
        self._l2_controllers = [
            L2Cache(
                self._l2_size,
                self._l2_assoc,
                self.ruby_system.network,
                self._num_l2_banks,
                cache_line_size,
            )
            for _ in range(self._num_l2_banks)
        ]
    ```

7. Create directory controllers for every memory port in the system

    ```python
    self._directory_controllers = [
        Directory(self.ruby_system.network, cache_line_size, range, port)
        for range, port in board.get_mem_ports()
    ]

    for dir in self._directory_controllers:
        dir.ruby_system = self.ruby_system
    ```


8. Connect the controllers in network and setup buffers 


 ```python
    if len(self._dma_controllers) != 0:
        self.ruby_system.dma_controllers = self._dma_controllers

    # Create the network and connect the controllers.
    self.ruby_system.network.connectControllers(
        self._l1_controllers
        + self._l2_controllers
        + self._directory_controllers
        + self._dma_controllers
    )
    self.ruby_system.network.setup_buffers()
```

