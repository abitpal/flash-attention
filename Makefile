# Makefile

NVCC = nvcc
PROFILER = nvprof
PROFILER_FLAGS = 
CFLAGS = -arch=sm_75 -O3 -lineinfo --use_fast_math --keep-device-functions -std=c++17
LIBS = -lcublas

# Directories
SRC_DIR = src
BUILD_DIR = build

# Source files (in src/)
SRCS = $(SRC_DIR)/test_flash_attention.cu

# Object files (in build/)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRCS))

# Executable target
TARGET = $(BUILD_DIR)/test_flash_attention

# Default rule
all: $(TARGET)
	./$(TARGET)

# Linking final binary
$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LIBS)

# Compile each .cu file to .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CFLAGS) -dc -o $@ $<

# Create build/ if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

profile: $(TARGET)
	$(PROFILER) $(PROFILER_FLAGS) $(TARGET)

clean:
	rm -f $(BUILD_DIR)/*.o $(TARGET)

.PHONY: all clean
