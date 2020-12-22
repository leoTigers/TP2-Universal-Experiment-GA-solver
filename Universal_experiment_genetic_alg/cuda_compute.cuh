#pragma once

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include "structs.h"
#include <curand.h>
#include <curand_kernel.h>

#define BLOCKS 100
#define THREADS 32
#define TOTAL_THREADS BLOCKS*THREADS

__global__ void cuda_breed(Parameters* d_params, Grid* d_population, int* d_scores,
    int* d_scores_indices, curandState* state);

void cuda_run(Parameters params, Parameters* d_params, Grid* population, Grid* d_population,
    int* scores, int* d_scores, int* scores_indices, int* d_scores_indices,
    Grid* fittest, Grid* d_fittest, curandState* state);

void setup(curandState* state, Parameters* d_params);

void quicksortIndices(int population_size, int values[], int indices[]);
void quicksortIndices(int population_size, int values[], int indices[], int low, int high);
void swap(int i, int j, int values[], int indices[]);