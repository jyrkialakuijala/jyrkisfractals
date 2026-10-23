# JyrkisFractals build. Toolchain locations can be overridden on the command line, e.g.
#   make CUDA_HOME=/usr/local/cuda ARCH=sm_89 SDL_LIBS="-lSDL2" GL_LIBS="-lGL"
CUDA_HOME ?= $(HOME)/.local/cuda13-venv/lib/python3.14/site-packages/nvidia/cu13
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CXX       ?= g++
ARCH      ?= sm_120

# Headers are vendored (third_party/include); libraries come from the system runtime packages.
SDL_LIBS ?= -l:libSDL2-2.0.so.0
GL_LIBS  ?= -l:libGL.so.1

INCLUDES  := -Isrc -Ithird_party/include -I$(CUDA_HOME)/include
CXXFLAGS  ?= -O2 -std=c++17 -Wall -Wextra
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -Wall
LDFLAGS   := -L$(CUDA_HOME)/lib -lcudart_static $(SDL_LIBS) $(GL_LIBS) -lz -ldl -lpthread -lrt

BUILD := build
TARGET := jyrkisfractals
OBJS := $(BUILD)/main.o $(BUILD)/image_io.o $(BUILD)/kernels.o

all: $(TARGET)

# Link to a temporary name and rename, so a running instance is never overwritten in place.
$(TARGET): $(OBJS)
	$(CXX) -o $@.tmp $^ $(LDFLAGS) && mv -f $@.tmp $@

$(BUILD)/%.o: src/%.cpp src/fractal.h src/image_io.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD)/%.o: src/%.cu src/fractal.h | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD) $(TARGET)

.PHONY: all run clean
