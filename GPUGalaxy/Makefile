NVCC := nvcc -arch sm_20 -g
NVCC_FLAGS := -g `sdl2-config --cflags --libs`

all: galaxyGPU

clean:
	@rm -f galaxyGPU

galaxyGPU: main.cu
	$(NVCC) $(NVCC_FLAGS) -o galaxyGPU main.cu

