# Makefile

NVCC = nvcc
CFLAGS = -arch=sm_70 -O3
LIBS = -lcublas

# Source and object files
SRCS = utils.cu flash_attention_kernel.cu test_flash_attention.cu
OBJS = utils.o flash_attention_kernel.o test_flash_attention.o

TARGET = test_flash_attention

# Default rule
all: $(TARGET)

# Linking final binary from object files
$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LIBS)

# Compile each .cu file to an object file with device code support
%.o: %.cu
	$(NVCC) $(CFLAGS) -dc -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
