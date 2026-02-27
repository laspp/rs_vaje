GEM5_PATH =/d/hpc/projects/FRI/GEM5/gem5_workspace/gem5/
ISA = riscv
GXX = riscv64-unknown-linux-gnu-gcc

all: matrix_mult.bin

matrix_mult.bin: matrix_mult.c
	$(GXX) -o matrix_mult.bin matrix_mult.c -static -fopenmp -lpthread -I$(GEM5_PATH)/include -L$(GEM5_PATH)/util/m5/build/$(ISA)/out -I$(GEM5_PATH)/util/m5/src -lm5 -DGEM5

clean:
	rm -f matrix_mult.bin