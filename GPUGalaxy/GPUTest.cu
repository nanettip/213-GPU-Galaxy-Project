#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>

#include <SDL.h>

#include "bitmap.hh"
#include "gui.hh"
#include "star.hh"
#include "util.hh"
#include "vec2d.hh"

__global__ void addx(vec2d* newX) {
  *newX += vec2d(1, 0);
}

int main() {

  vec2d center = vec2d(0, 0);
  vec2d* newX;
  
  if(cudaMalloc(&newX, sizeof(vec2d)) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate newcenter on GPU\n");
    exit(2);
  }

  if(cudaMemcpy(newX, &center, sizeof(vec2d), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy center to the GPU\n");
  }

  addx<<<3, 1>>>(newX);

  cudaDeviceSynchronize();

  if(cudaMemcpy(&center, newX, sizeof(vec2d), cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy newcenter from the GPU\n");
  }

  printf("%f\n", center.x());

  cudaFree(newX);

  return 0;
}
