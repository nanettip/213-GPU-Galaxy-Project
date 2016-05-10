#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>
#include <time.h>

#include <SDL.h>

#include "bitmap.hh"
#include "gui.hh"
#include "star.hh"
#include "util.hh"
#include "vec2d.hh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
using namespace std;

// Screen size
#define WIDTH 640
#define HEIGHT 480

// Minimum time between clicks
#define CREATE_INTERVAL 1000

// Time step size
#define DT 0.04

// Gravitational constant
#define G 1

// Update all stars in the simulation
void updateStars();


//CUDA compute forces and update all stars
__global__ void computeForce(double* mass, double* posX, double* posY,
                             double* forceX, double* forceY, int starSize,
                             double* velX, double* velY, double* radius,
                             int* merge, int* initialized);

//CUDA update star's positions
__global__ void updateStar(double* mass, double* posX, double* posY,
                           double* prev_posX, double* prev_posY, double* forceX,
                           double* forceY, double* velX, double* velY,
                           int* initialized, int starSize, int* merge);

// Draw a circle on a bitmap based on this star's position and radius
void drawStar(bitmap* bmp, star s);

// Add a "galaxy" of stars to the points list
void addRandomGalaxy(double center_x, double center_y);



// A list of stars being simulated
vector<star> stars;

// Offset of the current view
int x_offset = 0;
int y_offset = 0;

/**
 * Entry point for the program
 * \param argc  The number of command line arguments
 * \param argv  An array of command line arguments
 */

// Keep track of how many stars are shown
int starSize;


int main(int argc, char** argv) {
  // Seed the random number generator
  srand(time(NULL));
  
  // Create a GUI window
  gui ui("Galaxy Simulation", WIDTH, HEIGHT);
  
  // Start with the running flag set to true
  bool running = true;
  
  // Render everything using this bitmap
  bitmap bmp(WIDTH, HEIGHT);
  
  // Save the last time the mouse was clicked
  bool mouse_up = true;


  
  // Keep track of how many galaxies were made
  int galaxies = 0;

  // Count the number of frames
  int numOfFrames = 0;
  time_t startTime;


  
  // Loop until we get a quit event
  while(running) {


    
    // Increment number of frames
    numOfFrames++;
    // Keep track of runtime
    startTime = time_ms();


    
    // Process events
    SDL_Event event;
    while(SDL_PollEvent(&event) == 1) {
      // If the event is a quit event, then leave the loop
      if(event.type == SDL_QUIT) running = false;
    }
    
    // Get the current mouse state
    int mouse_x, mouse_y;
    uint32_t mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);
    
    // If the left mouse button is pressed, create a new random "galaxy"
    if(mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
      // Only create one if the mouse button has been released
      if(mouse_up) {
        addRandomGalaxy(mouse_x - x_offset, mouse_y - y_offset);
        galaxies++; // Increment galaxy count
        // Don't create another one until the mouse button is released
        mouse_up = false;
      }
    } else {
      // The mouse button was released
      mouse_up = true;
    }
    
    // Get the keyboard state
    const uint8_t* keyboard = SDL_GetKeyboardState(NULL);
    
    // If the up key is pressed, shift up one pixel
    if(keyboard[SDL_SCANCODE_UP]) {
      y_offset++;
      bmp.shiftDown();  // Shift pixels so scrolling doesn't create trails
    }
    
    // If the down key is pressed, shift down one pixel
    if(keyboard[SDL_SCANCODE_DOWN]) {
      y_offset--;
      bmp.shiftUp();  // Shift pixels so scrolling doesn't create trails
    }
    
    // If the right key is pressed, shift right one pixel
    if(keyboard[SDL_SCANCODE_RIGHT]) {
      x_offset--;
      bmp.shiftLeft();  // Shift pixels so scrolling doesn't create trails
    }
    
    // If the left key is pressed, shift left one pixel
    if(keyboard[SDL_SCANCODE_LEFT]) {
      x_offset++;
      bmp.shiftRight(); // Shift pixels so scrolling doesn't create trails
    }
    
    // Remove stars that have NaN positions
    for(int i=0; i<stars.size(); i++) {
      // Remove this star if it is too far from zero or has NaN position
      if(stars[i].pos().x() != stars[i].pos().x() ||  // A NaN value does not equal itself
         stars[i].pos().y() != stars[i].pos().y()) {
        stars.erase(stars.begin()+i);
        i--;
        continue;
      }
    }
    // Update star count
    starSize = stars.size();

    // Create arrays for the star attributes
    // This is needed to compute forces and update stars on GPU
    double starMass[starSize];
    double starPosX[starSize];
    double starPrevX[starSize];
    double starForceX[starSize];
    double starVelX[starSize];
    double starPosY[starSize];
    double starPrevY[starSize];
    double starForceY[starSize];
    double starVelY[starSize];    
    int starInit[starSize];
    double starRad[starSize];
    int starMerge[starSize]; //Lets us know if a star has been merged.

    // Assign values to arrays.
    for(int i=0; i <starSize; i++) {
      starMass[i] = stars[i].mass();
      starPosX[i] = stars[i].pos().x();
      starPrevX[i] = stars[i].prev_pos().x();
      starForceX[i] = stars[i].force().x();
      starVelX[i] = stars[i].vel().x();
      starPosY[i] = stars[i].pos().y();
      starPrevY[i] = stars[i].prev_pos().y();
      starForceY[i] = stars[i].force().y();
      starVelY[i] = stars[i].vel().y();
      starInit[i] = stars[i].initialized();
      starRad[i] = stars[i].radius();
      starMerge[i] = i;
    }

    // Create empty arrays for GPU
    double* starMassGPU;
    double* starPosXGPU;
    double* starPrevXGPU;
    double* starForceXGPU;
    double* starVelXGPU;
    double* starPosYGPU;
    double* starPrevYGPU;
    double* starForceYGPU;
    double* starVelYGPU;    
    int* starInitGPU;
    double* starRadGPU;
    int* starMergeGPU;
    
    // Malloc all GPU arrays
    
    if(cudaMalloc(&starMassGPU, sizeof(double) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starMassGPU on GPU\n");
        exit(2);
      }
    if(cudaMalloc(&starPosXGPU, sizeof(double) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starPosXGPU on GPU\n");
        exit(2);
      }
    if(cudaMalloc(&starPrevXGPU, sizeof(double) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starPrevXGPU on GPU\n");
        exit(2);
      }
    if(cudaMalloc(&starForceXGPU, sizeof(double) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starForceXGPU on GPU\n");
        exit(2);
      }
    if(cudaMalloc(&starVelXGPU, sizeof(double) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starVelXGPU on GPU\n");
        exit(2);
      }
    if(cudaMalloc(&starPosYGPU, sizeof(double) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starPosYGPU on GPU\n");
        exit(2);
      }
    if(cudaMalloc(&starPrevYGPU, sizeof(double) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starPrevYGPU on GPU\n");
        exit(2);
      }  
    if(cudaMalloc(&starForceYGPU, sizeof(double) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starForceYGPU on GPU\n");
        exit(2);
      }    
    if(cudaMalloc(&starVelYGPU, sizeof(double) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starVelYGPU on GPU\n");
        exit(2);
      }
    if(cudaMalloc(&starInitGPU, sizeof(int) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starInitGPU on GPU\n");
        exit(2);
      }
    if(cudaMalloc(&starRadGPU, sizeof(double) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starRadGPU on GPU\n");
        exit(2);
      }
    if(cudaMalloc(&starMergeGPU, sizeof(int) * (starSize)) != cudaSuccess)
      {
        fprintf(stderr, "Failed to allocate starGPUGPU on GPU\n");
        exit(2);
      }

    
    //Copy the host data to device
    
    if(cudaMemcpy(starMassGPU, starMass, sizeof(double) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starMassGPU to the GPU");
      }
    if(cudaMemcpy(starPosXGPU, starPosX, sizeof(double) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starPosXGPU to the GPU");
      }
    if(cudaMemcpy(starPrevXGPU, starPrevX, sizeof(double) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starPrevXGPU to the GPU");
      }
    if(cudaMemcpy(starForceXGPU, starForceX, sizeof(double) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starForceXGPU to the GPU");
      }
    if(cudaMemcpy(starVelXGPU, starVelX, sizeof(double) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starVelXGPU to the GPU");
      }
    if(cudaMemcpy(starPosYGPU, starPosY, sizeof(double) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starPosYGPU to the GPU");
      }
    if(cudaMemcpy(starPrevYGPU, starPrevY, sizeof(double) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starPrevYGPU to the GPU");
      }
           
    if(cudaMemcpy(starForceYGPU, starForceY, sizeof(double) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starForceYGPU to the GPU");
      }
    if(cudaMemcpy(starVelYGPU, starVelY, sizeof(double) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starVelYGPU to the GPU");
      }
    if(cudaMemcpy(starInitGPU, starInit, sizeof(int) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starInitGPU to the GPU");
      }
    if(cudaMemcpy(starRadGPU, starRad, sizeof(double) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starRadGPU to the GPU");
      }
    if(cudaMemcpy(starMergeGPU, starMerge, sizeof(int) * (starSize),
                  cudaMemcpyHostToDevice)
       != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starMergeGPU to the GPU");
      }

    // Compute forces on GPU with blocks for each star
    computeForce<<<starSize,1>>>
      (starMassGPU, starPosXGPU, starPosYGPU, starForceXGPU, starForceYGPU,
       starSize, starVelXGPU, starVelYGPU, starRadGPU, starMergeGPU,
       starInitGPU);
     
    cudaDeviceSynchronize();

    // Update star positions on GPU with blocks for each star
    updateStar<<<starSize,1>>>(starMassGPU, starPosXGPU, starPosYGPU,
                               starPrevXGPU, starPrevYGPU, starForceXGPU,
                               starForceYGPU, starVelXGPU, starVelYGPU,
                               starInitGPU, starSize, starMergeGPU);
    
    cudaDeviceSynchronize();


    // Copy the device data to the host
    
    if(cudaMemcpy(&starMass, starMassGPU, sizeof(double) * (starSize),
                  cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starMassGPU to the CPU\n");
      }
    if(cudaMemcpy(&starPosX, starPosXGPU, sizeof(double) * (starSize),
                  cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starPosXGPU to the CPU\n");
      }
    if(cudaMemcpy(&starPrevX, starPrevXGPU, sizeof(double) * (starSize),
                  cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starPrevXGPU to the CPU\n");
      }
    if(cudaMemcpy(&starVelX, starVelXGPU, sizeof(double) * (starSize),
                  cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starVelXGPU to the CPU\n");
      }
    if(cudaMemcpy(&starPosY, starPosYGPU, sizeof(double) * (starSize),
                  cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starPosYGPU to the CPU\n");
      }
    if(cudaMemcpy(&starPrevY, starPrevYGPU, sizeof(double) * (starSize),
                  cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starPrevYGPU to the CPU\n");
      }
    if(cudaMemcpy(&starVelY, starVelYGPU, sizeof(double) * (starSize),
                  cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starVelYGPU to the CPU\n");
      }
    if(cudaMemcpy(&starInit, starInitGPU, sizeof(int) * (starSize),
                  cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starInitGPU to the CPU\n");
      }
    if(cudaMemcpy(&starRad, starRadGPU, sizeof(double) * (starSize),
                  cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starRadGPU to the CPU\n");
      }
    if(cudaMemcpy(&starMerge, starMergeGPU, sizeof(int) * (starSize),
                  cudaMemcpyDeviceToHost) != cudaSuccess)
      {
        fprintf(stderr, "Failed to copy starInitGPU to the CPU\n");
      }   

    // Free the arrays on the GPU
    
    cudaFree(starMassGPU);
    cudaFree(starPosXGPU);
    cudaFree(starPrevXGPU);
    cudaFree(starVelXGPU);
    cudaFree( starForceXGPU);
    cudaFree(starPosYGPU);
    cudaFree(starPrevYGPU);
    cudaFree(starForceYGPU);
    cudaFree(starVelYGPU);    
    cudaFree(starInitGPU);
    cudaFree(starMergeGPU);
    cudaFree(starRadGPU);

    // Update the stars in the vector with the data from the arrays
    // Also remove the stars that were merged
    int displacement = 0;
    int j = 0;
    for(int i=0; i <(starSize - displacement); i++) {
      if(starMerge[j] == j) {
        stars[i].changeMass(starMass[j]);
        stars[i].changePos(vec2d(starPosX[j], starPosY[j]));
        stars[i].changePrev(vec2d(starPrevX[j], starPrevY[j]));
        stars[i].changeVel(vec2d(starVelX[j], starVelY[j]));
        stars[i].changeInit(starInit[j]);
      }// if
      else {
        stars.erase(stars.begin() + i);
        i--;
        displacement++;
      }// else
      j++;
    }// fpr

    // Darken the bitmap instead of clearing it to leave trails
    bmp.darken(0.92);
    
    // Draw stars
    for(int i=0; i<stars.size(); i++) {
      drawStar(&bmp, stars[i]);
    }// for
    
    // Display the rendered frame
    ui.display(bmp);

    // Calculate the run time
    time_t endTime = time_ms();
    time_t elapsedTime = endTime - startTime;

    // Print frame data for 2000 frames
    if (numOfFrames < 2000)
      printf("%d, %d, %lu\n", galaxies, numOfFrames, elapsedTime);
  }// while
  
  return 0;
}// main


// Compute the force on the star based on the forces on the other stars
// This is done on the GPU
__global__ void computeForce(double* mass, double* posX, double* posY,
                             double* forceX, double* forceY, int starSize,
                             double* velX, double* velY, double* radius,
                             int* merge, int* initialized){
  // If the star has not been merged
  if(merge[blockIdx.x] == blockIdx.x) {
    double m1 = mass[blockIdx.x];
    vec2d pos = vec2d(posX[blockIdx.x], posY[blockIdx.x]);
    vec2d vel = vec2d(velX[blockIdx.x], velY[blockIdx.x]);
    // Loop on all other stars
    for(int j = blockIdx.x + 1; j<starSize; j++) {
      // If the current star hasn't been merged
      if(merge[j] == j) {
        double m2 = mass[j];
        vec2d pos2 = vec2d(posX[j], posY[j]);
        vec2d vel2 = vec2d(velX[j], velY[j]);

        // Compute a vector between two points
        vec2d diff = pos - pos2;

        // Compute the distance between the two points
        double dist = diff.magnitude();
        
        // If the objects are too close, merge them
        if(dist < (radius[blockIdx.x] + radius[j]) / 1.5) {
          merge[j] = blockIdx.x;
          mass[blockIdx.x] = m1 + m2;
          pos = (pos * m1 + pos2 * m2) / (m1 + m2);
          vel = (vel * m1 + vel2 * m2) / (m1 + m2);
          m1 = mass[blockIdx.x];
          posX[blockIdx.x] = pos.x();
          posY[blockIdx.x] = pos.y();
          velX[blockIdx.x] = vel.x();
          velY[blockIdx.x] = vel.y();
          forceX[blockIdx.x] = 0;
          forceY[blockIdx.x] = 0;
          initialized[blockIdx.x] = 0;
        }// if
        else{
          // Normalize the difference vector to be a unit vector
          diff = diff.normalized();

          // Compute the force between these two stars
          vec2d force = -diff * G * m1 * m2 / (dist * dist);

          // Apply the force to both stars
          forceX[blockIdx.x] += force.x();
          forceY[blockIdx.x] += force.y();
          forceX[j] += (-force.x());
          forceY[j] += (-force.y());
        }// else
      }// if
    }// for
  }// if
}// computeForce()

// Update the positions of the star based on the forces acting on it.
// This is done on the GPU
__global__ void updateStar(double* mass, double* posX, double* posY,
                           double* prev_posX, double* prev_posY,
                           double* forceX, double* forceY, double* velX,
                           double* velY, int* initialized, int starSize,
                           int* merge){
  // Check to see if star has not been merged
  if(merge[blockIdx.x] == blockIdx.x) {
    vec2d pos = vec2d(posX[blockIdx.x], posY[blockIdx.x]);
    vec2d prev_pos = vec2d(prev_posX[blockIdx.x], prev_posY[blockIdx.x]);
    vec2d force = vec2d(forceX[blockIdx.x], forceY[blockIdx.x]);
    vec2d vel = vec2d(velX[blockIdx.x], velY[blockIdx.x]);
  
    vec2d accel = force / mass[blockIdx.x];

    // Verlet integration
    if(initialized[blockIdx.x] == 0) { // First step: no previous position
      vec2d next_pos = pos + vel * DT + accel / 2 * DT * DT;
      prev_pos = pos;
      pos = next_pos;
      initialized[blockIdx.x] = 1;
    }// if
    else { // Later steps
      vec2d next_pos = pos * 2 - prev_pos + accel * DT * DT;
      prev_pos = pos;
      pos = next_pos;
    }// else

    posX[blockIdx.x] = pos.x();
    posY[blockIdx.x] = pos.y();
    prev_posX[blockIdx.x] = prev_pos.x();
    prev_posY[blockIdx.x] = prev_pos.y();

    // Track velocity, even though this isn't strictly required
    vel += accel * DT;
    velX[blockIdx.x] = vel.x();
    velY[blockIdx.x] = vel.y();

    // Zero out the force
    forceX[blockIdx.x] = 0;
    forceY[blockIdx.x] = 0;
  }// if
}// updateStar()



// Create a circle of stars moving in the same direction around the center of mass
void addRandomGalaxy(double center_x, double center_y) {
  // Random number of stars
  int count = rand() % 1000 + 1000;
  
  // Random radius
  double radius = drand(50, 200);
  
  // Create a vector for the center of the galaxy
  vec2d center = vec2d(center_x, center_y);
  
  // Clockwise or counter-clockwise?
  double direction = 1;
  if(rand() % 2 == 0) direction = -1;
  
  // Create `count` stars
  for(int i=0; i<count; i++) {
    // Generate a random angle
    double angle = drand(0, M_PI * 2);
    // Generate a random radius, biased toward the center
    double point_radius = drand(0, sqrt(radius)) * drand(0, sqrt(radius));
    // Compute X and Y coordinates
    double x = point_radius * sin(angle);
    double y = point_radius * cos(angle);
    
    // Create a vector to hold the position of this star (origin at center of the "galaxy")
    vec2d pos = vec2d(x, y);
    // Move the star in the appropriate direction around the center, with slightly-random velocity
    vec2d vel = vec2d(-cos(angle), sin(angle)) * sqrt(point_radius) * direction * drand(0.25, 1.25);
    
    // Create a new random color for the star
    rgb32 color = rgb32(rand() % 64 + 192, rand() % 64 + 192, 128);
    
    // Add the star with a mass dependent on distance from the center of the "galaxy"
    stars.push_back(star(10 / sqrt(pos.magnitude()), pos + center, vel, color));
  }
}

// Draw a circle at the given star's position
// Uses method from http://groups.csail.mit.edu/graphics/classes/6.837/F98/Lecture6/circle.html
void drawStar(bitmap* bmp, star s) {
  double center_x = s.pos().x();
  double center_y = s.pos().y();
  double radius = s.radius();
  
  // Loop over points in the upper-right quadrant of the circle
  for(double x = 0; x <= radius*1.1; x++) {
    for(double y = 0; y <= radius*1.1; y++) {
      // Is this point within the circle's radius?
      double dist = sqrt(pow(x, 2) + pow(y, 2));
      if(dist < radius) {
        // Set this point, along with the mirrored points in the other three quadrants
        bmp->set(center_x + x + x_offset, center_y + y + y_offset, s.color());
        bmp->set(center_x + x + x_offset, center_y - y + y_offset, s.color());
        bmp->set(center_x - x + x_offset, center_y - y + y_offset, s.color());
        bmp->set(center_x - x + x_offset, center_y + y + y_offset, s.color());
      }
    }
  }
}
