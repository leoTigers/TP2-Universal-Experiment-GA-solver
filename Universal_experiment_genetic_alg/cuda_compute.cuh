#pragma once

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include "structs.h"

void ce(int population_size, Grid* population, Grid* d_population);
