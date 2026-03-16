GEM5_PATH = /d/hpc/projects/FRI/GEM5/gem5_workspace/gem5/
ISA = riscv
GXX = riscv64-linux-gnu-gcc


all: work_imbalanced.bin

work_imbalanced.bin: work_imbalanced.c
	$(GXX) -o work_imbalanced.bin work_imbalanced.c  -static  -fopenmp -lpthread -I$(GEM5_PATH)/include -L$(GEM5_PATH)/util/m5/build/$(ISA)/out -I$(GEM5_PATH)/util/m5/src -lm5 -DGEM5

clean:
	rm -f work_imbalanced.bin