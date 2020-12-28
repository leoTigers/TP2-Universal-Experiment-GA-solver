#include "cuda_compute.cuh"

void repr(Grid& individual);
void create_population(int population_size, Grid* population);
void copy_tab(Grid& g1, Grid& g2);

__global__ void cuda_eval(Parameters* d_params, Grid* population, int* scores) {
    
    int indice = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (indice >= d_params->population_size)
        return;
    //if grid hasen't changed
    if (!population[indice].changed)
    {
        scores[indice] = population[indice].score;
        return;
    }


    char movements[8][2] = { {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0} };
    // starting pos
    char x = -1, y = 6;
    char cur_dir = 3;

    bool end = false;
    int score = 0;
    char uses;
    char type;
    char dir;
    int cc = 0;

    //copy_tab(individual, population[indice]);
    for (int i = 0; i < DIM; ++i){
        for (int j = 0; j < DIM; ++j) {
            population[indice].cases[i][j].current_uses = 0;
            population[indice].cases[i][j].c_dir = population[indice].cases[i][j].dir;
        }
    }

    do {
        cc++;
        if (cc > 50000) {
            scores[indice] = 50000;
            return;
        }
        //repr(population[indice]);
        //move
        x += movements[cur_dir][0];
        y += movements[cur_dir][1];

        //check exit cond
        if (x < 0 || x >= DIM || y < 0 || y >= DIM) {
            end = true;
        }
        else {
            switch (population[indice].cases[x][y].type) {
            case Object_type::ARROW:
                type = population[indice].cases[x][y].o_type;
                uses = population[indice].cases[x][y].current_uses;
                dir = population[indice].cases[x][y].c_dir;
                if ((type == Arrow_type::ONE_USE && uses < 1) || (type == Arrow_type::THREE_USES && uses < 3) || (type == Arrow_type::FIVE_USES && uses < 5)) {
                    population[indice].cases[x][y].current_uses++;
                    cur_dir = dir;
                }
                else if (type == Arrow_type::INFINITE_USES) {
                    cur_dir = dir;
                }
                else if (type == Arrow_type::ROTATING) {
                    cur_dir = dir;
                    population[indice].cases[x][y].c_dir = (dir + 1) & 7;
                }
                break;
            case Object_type::ORB:
                switch (population[indice].cases[x][y].o_type)
                {
                case Orb_type::NORMAL:
                    score--;
                    break;
                case Orb_type::REFRESH:
                    if (population[indice].cases[x][y].current_uses == 0) { //refresh not used
                        for (int i = 0; i < DIM; ++i) {
                            for (int j = 0; j < DIM; ++j) {
                                population[indice].cases[i][j].current_uses = 0;
                                //population[indice].cases[i][j].c_dir = population[indice].cases[i][j].dir;
                            }
                        }
                        population[indice].cases[x][y].current_uses = 1;
                    }
                    break;
                }
                break;
            case Object_type::REFLECT: //reflect
                uses = population[indice].cases[x][y].current_uses;
                if (uses < 2) {
                    uses++;
                    if (cur_dir & 1)
                        cur_dir = (cur_dir + 4) % 8;
                    else if (cur_dir == 0)
                        cur_dir = 6;
                    else
                        cur_dir -= 2;

                    population[indice].cases[x][y].current_uses++;
                }
                break;
            }
        }
    } while (!end);
    population[indice].score = score;
    population[indice].changed = false;
    scores[indice] = score;
    //printf("%d\t%d\n", indice, score);
}

__device__ void create_object(Parameters* d_param, Case* object, char limits[], 
    curandState* state, int index, int x, int y) {

    curandState localState = state[index];

    char type = curand(&localState) % (REFLECT_UNLOCK ? 3 : 2);
    object->type = type;
    switch (type) {
    case Object_type::ARROW: //arrow
        // pick an arrow type within grid object limits
        do {
            object->o_type = curand(&localState) % 5;
        } while ((object->o_type == Arrow_type::FIVE_USES && limits[0] >= MAX_5T_ARROWS) ||
            (object->o_type == Arrow_type::INFINITE_USES && limits[1] >= MAX_INF_ARROWS) ||
            (object->o_type == Arrow_type::ROTATING && limits[2] >= MAX_ROT_ARROWS));

        if (object->o_type > 1)
            limits[object->o_type - 2]++;

        if (x == 0) {
            if (y == 0)
                object->dir = curand(&localState) % 3 + 3;
            else if(y==DIM-1)
                object->dir = curand(&localState) % 3 + 1;
            else
                object->dir = curand(&localState) % 5 + 1;
        }
        else if (x == DIM - 1) {
            if (y == 0)
                object->dir = curand(&localState) % 3 + 5;
            else if (y == DIM - 1)
                object->dir = (curand(&localState) % 3 + 6)&7;
            else
                object->dir = (curand(&localState) % 5 + 5)&7;

        }
        else if (y == 0) {
            object->dir = curand(&localState) % 5 + 3;
        }
        else if (y == DIM - 1) {
            object->dir = (curand(&localState) % 5 + 7)&7;
        }
        
        //object->dir = curand(&localState) & 7;
        break;
    case Object_type::ORB: //orb
        // orb can be normal or refresh
        object->o_type = (limits[3] < MAX_REFRESH) ? curand(&localState) & 1 : 0;
        limits[3] += (object->o_type & 1);
        break;
    }
    state[index] = localState;
}

__global__ void cuda_breed(Parameters *d_params, Grid* d_population, int* d_scores, 
    int* d_scores_indices, curandState* state) {

    int indice = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (indice >= d_params->population_size)
        return;

    if (indice < d_params->retain_rate * d_params->population_size)
        return;

    curandState localState = state[d_scores_indices[indice]];

    int parent_a, parent_b;
    int smod = d_params->retain_rate * d_params->population_size;
    int p_a, p_b;
    p_a = (int)(powf(curand_uniform(&localState), 5)*smod);
    p_b = (int)(powf(curand_uniform(&localState), 5)*smod);
    parent_a = d_scores_indices[p_a];
    parent_b = d_scores_indices[p_b];

    // Reset limits for new individual
    for (int i = 0; i < 4; i++)
        d_population[d_scores_indices[indice]].limits[i] = 0;

    int crosspoint = curand(&localState) % 50; // the case at which point we'll take values from parent_b instead of parent_a
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            Case tmp;
            if (i * DIM + j < crosspoint) {
                // Normal addition by crosspoint
                tmp = d_population[parent_a].cases[i][j];
                switch (tmp.type) {
                case Object_type::ARROW:
                    if (tmp.o_type > 1)
                        d_population[d_scores_indices[indice]].limits[tmp.o_type - 2]++;
                    break;
                case Object_type::ORB:
                    if (tmp.o_type == Orb_type::REFRESH)
                        d_population[d_scores_indices[indice]].limits[3]++;
                    break;
                }
            }
            else { // Addition with limit check
                tmp = d_population[parent_b].cases[i][j];
                int valid_insertion = true;
                switch (tmp.type) {
                case Object_type::ARROW:
                    switch (tmp.o_type)
                    {
                    case Arrow_type::FIVE_USES:
                        if (d_population[d_scores_indices[indice]].limits[0] >= MAX_5T_ARROWS)
                            valid_insertion = false;
                        break;
                    case Arrow_type::INFINITE_USES:
                        if (d_population[d_scores_indices[indice]].limits[1] >= MAX_INF_ARROWS)
                            valid_insertion = false;
                        break;
                    case Arrow_type::ROTATING:
                        if (d_population[d_scores_indices[indice]].limits[2] >= MAX_ROT_ARROWS)
                            valid_insertion = false;
                        break;
                    default:
                        break;
                    }
                    if (!valid_insertion) {
                        create_object(d_params, &tmp, d_population[d_scores_indices[indice]].limits,
                            state, indice, i , j);
                    }
                    else {
                        if (tmp.o_type > 1)
                            d_population[d_scores_indices[indice]].limits[tmp.o_type - 2]++;
                    }
                    break;
                case Object_type::ORB:
                    if (tmp.o_type == Orb_type::REFRESH) {
                        if (d_population[d_scores_indices[indice]].limits[3] >= MAX_REFRESH)
                        {
                            create_object(d_params, &tmp, d_population[d_scores_indices[indice]].limits,
                                state, indice, i, j);
                        }
                        else {
                            d_population[d_scores_indices[indice]].limits[3]++;
                        }
                    }
                    break;
                }
            }
            d_population[d_scores_indices[indice]].cases[i][j] = tmp;
        }
    }
    d_population[d_scores_indices[indice]].changed = true;
    state[d_scores_indices[indice]] = localState;
}


__global__ void cuda_mutate(Parameters* d_params, Grid* d_population, curandState* state,
    int *d_scores_indices) {
    int indice = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (indice >= d_params->population_size || indice == 0)
        return;
    
    if (indice < d_params->retain_rate * d_params->population_size)
        return;


    curandState localState = state[d_scores_indices[indice]];

    if (curand_uniform(&localState) > d_params->mutation_rate)
    {
        state[d_scores_indices[indice]] = localState;
        return;
    }

    unsigned int mut_count = powf(curand_uniform(&localState),3) * (d_params->max_mutations-d_params->min_mutations) + d_params->min_mutations;

    for (int i = 0; i < mut_count; ++i) {
        if (curand_uniform(&localState) > 0.7) { // exchange 2 cells
            int x1, y1, x2, y2;
            x1 = curand(&localState) % DIM;
            y1 = curand(&localState) % DIM;
            x2 = curand(&localState) % DIM;
            y2 = curand(&localState) % DIM;
            Case tmp = d_population[d_scores_indices[indice]].cases[x1][y1];
            d_population[d_scores_indices[indice]].cases[x1][y1] = d_population[d_scores_indices[indice]].cases[x2][y2];
            d_population[d_scores_indices[indice]].cases[x2][y2] = tmp;
        }
        else { // mutate a cell
            // pick a random position
            int x = curand(&localState) % DIM;
            int y = curand(&localState) % DIM;

            int type;

            // update limits
            switch (d_population[d_scores_indices[indice]].cases[x][y].type) {
            case Object_type::ARROW:
                type = d_population[d_scores_indices[indice]].cases[x][y].o_type;
                switch (type) {
                case Arrow_type::FIVE_USES:
                    d_population[d_scores_indices[indice]].limits[0]--;
                    break;
                case Arrow_type::INFINITE_USES:
                    d_population[d_scores_indices[indice]].limits[1]--;
                    break;
                case Arrow_type::ROTATING:
                    d_population[d_scores_indices[indice]].limits[2]--;
                    break;
                }
                break;
            case Object_type::ORB:
                if (d_population[d_scores_indices[indice]].cases[x][y].o_type == Orb_type::REFRESH)
                    d_population[d_scores_indices[indice]].limits[3]--;
                break;
            }

            // create new object
            create_object(d_params, &d_population[d_scores_indices[indice]].cases[x][y], d_population[d_scores_indices[indice]].limits,
                state, indice, x, y);
        }
    }
    
    //for (int i = 0; i < d_params->max_mutations-d_params->min_mutations; ++i) {
    //    if (curand_uniform(&localState) < d_params->mutation_rate) {
    //        // pick a random position
    //        int x = curand(&localState) % DIM;
    //        int y = curand(&localState) % DIM;
    //       
    //        int type;
    //        // update limits
    //        switch (d_population[d_scores_indices[indice]].cases[x][y].type) {
    //        case Object_type::ARROW:
    //            type = d_population[d_scores_indices[indice]].cases[x][y].o_type;
    //            switch (type) {
    //            case Arrow_type::FIVE_USES:
    //                d_population[d_scores_indices[indice]].limits[0]--;
    //                break;
    //            case Arrow_type::INFINITE_USES:
    //                d_population[d_scores_indices[indice]].limits[1]--;
    //                break;
    //            case Arrow_type::ROTATING:
    //                d_population[d_scores_indices[indice]].limits[2]--;
    //                break;
    //            }
    //            break;
    //        case Object_type::ORB:
    //            if (d_population[d_scores_indices[indice]].cases[x][y].o_type == Orb_type::REFRESH)
    //                d_population[d_scores_indices[indice]].limits[3]--;
    //            break;
    //        }
    //
    //        // create new object
    //        type = curand(&localState) % (REFLECT_UNLOCK ? 3 : 2);
    //        d_population[d_scores_indices[indice]].cases[x][y].type = type;
    //        switch (type) {
    //        case Object_type::ARROW: //arrow
    //            // pick an arrow type within grid object limits
    //            do {
    //                d_population[d_scores_indices[indice]].cases[x][y].o_type = curand(&localState) % 5;
    //            } while ((d_population[d_scores_indices[indice]].cases[x][y].o_type == Arrow_type::FIVE_USES && d_population[d_scores_indices[indice]].limits[0] == MAX_5T_ARROWS) ||
    //                (d_population[d_scores_indices[indice]].cases[x][y].o_type == Arrow_type::INFINITE_USES && d_population[d_scores_indices[indice]].limits[1] == MAX_INF_ARROWS) ||
    //                (d_population[d_scores_indices[indice]].cases[x][y].o_type == Arrow_type::ROTATING && d_population[d_scores_indices[indice]].limits[2] == MAX_ROT_ARROWS));
    //
    //            if (d_population[d_scores_indices[indice]].cases[x][y].o_type > 1)
    //                d_population[d_scores_indices[indice]].limits[d_population[d_scores_indices[indice]].cases[x][y].o_type - 2]++;
    //            d_population[d_scores_indices[indice]].cases[x][y].dir = curand(&localState) & 7;
    //            break;
    //        case Object_type::ORB: //orb
    //            // orb can be normal or refresh
    //            d_population[d_scores_indices[indice]].cases[x][y].o_type = (d_population[d_scores_indices[indice]].limits[3] < MAX_REFRESH) ? curand(&localState) & 1 : 0;
    //            d_population[d_scores_indices[indice]].limits[3] += (d_population[d_scores_indices[indice]].cases[x][y].o_type & 1);
    //            break;
    //        }
    //        d_population[d_scores_indices[indice]].changed = true;
    //    }
    //}
    d_population[d_scores_indices[indice]].changed = true;
    state[d_scores_indices[indice]] = localState;
}

void quicksortIndices(int population_size, int values[], int indices[]) {
    for (int i = 0; i < population_size; i++)
        indices[i] = i;
    quicksortIndices(population_size, values, indices, 0, population_size - 1);
}

/**
 * @brief Sorts the provided values between two indices while applying the same
 *        transformations to the array of indices
 *
 * @param values  the values to sort
 * @param indices the indices to sort according to the corresponding values
 * @param         low, high are the **inclusive** bounds of the portion of array
 *                to sort
 */
void quicksortIndices(int population_size, int values[], int indices[], int low, int high) {
    int l = low;
    int h = high;
    int pivot = values[l];
    while (l <= h) {
        if (values[l] < pivot)
            l++;
        else if (values[h] > pivot)
            h--;
        else {
            swap(l, h, values, indices);
            l++;
            h--;
        }
    }
    if (low < h)
        quicksortIndices(population_size, values, indices, low, h);
    if (high > l)
        quicksortIndices(population_size, values, indices, l, high);
}

/**
 * @brief Swaps the elements of the given arrays at the provided positions
 *
 * @param         i, j the indices of the elements to swap
 * @param values  the array floats whose values are to be swapped
 * @param indices the array of ints whose values are to be swapped
 */
void swap(int i, int j, int values[], int indices[]) {
    int tempValue = values[i];
    int tempIndice = indices[i];
    values[i] = values[j];
    indices[i] = indices[j];
    values[j] = tempValue;
    indices[j] = tempIndice;
}

__global__ void setup_kernel(curandState* state, int* d_val) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(d_val[id], id, 0, &state[id]);
}

void setup(curandState* state, Parameters *params) {
    int  *values, *d_values;
    values = new int[(params->population_size / THREADS + 1)];
    cudaMalloc((void**)&d_values, sizeof(int) * (params->population_size / THREADS + 1));
    for (int i = 0; i < (params->population_size / THREADS + 1); ++i)
        values[i] = rand();
    cudaMemcpy(d_values, values, sizeof(int) * (params->population_size / THREADS + 1), cudaMemcpyHostToDevice);
    setup_kernel<<<(params->population_size / THREADS + 1), THREADS>>>(state, d_values);
    cudaFree(d_values);
    delete values;
}

void cuda_run(Parameters params, Parameters* d_params, Grid* population, Grid* d_population, 
    int* scores, int *d_scores, int* scores_indices, int *d_scores_indices, 
    Grid *fittest, Grid *d_fittest, curandState* state) {
    
    // evaluate the population
    cuda_eval<<<(params.population_size / THREADS + 1), THREADS>>>(d_params, d_population, d_scores);
    cudaDeviceSynchronize();

    // get back the scores
    cudaMemcpy(scores, d_scores, sizeof(int) * params.population_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < params.population_size; ++i) {
        scores_indices[i] = i;
        if (scores[i] == 50000) {
            cudaMemcpy(fittest, &d_population[i], sizeof(Grid), cudaMemcpyDeviceToHost);
            std::cout << "ERROR" << std::endl;
            repr(*fittest);
        }
    }
    quicksortIndices(params.population_size, scores, scores_indices);
    
    //save the fittest and his score
    if (scores[0] < fittest->score) {
        cudaMemcpy(fittest, &d_population[scores_indices[0]], sizeof(Grid), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(d_scores, scores, sizeof(int) * params.population_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores_indices, scores_indices, sizeof(int) * params.population_size, cudaMemcpyHostToDevice);

    // breed new elements
    cuda_breed<<<(params.population_size / THREADS + 1), THREADS>>>(d_params, d_population, d_scores, d_scores_indices, state);
    cudaDeviceSynchronize(); 

    // mutate some elements
    cuda_mutate<<<(params.population_size / THREADS + 1), THREADS>>>(d_params, d_population, state, d_scores_indices);
    cudaError a = cudaDeviceSynchronize();
    //cudaMemcpy(population, d_population, sizeof(Grid) * params.population_size, cudaMemcpyDeviceToHost);

    //cudaFree(d_population);
    //cudaFree(d_population_size);
}

int benchmark(Parameters params, int verbose) {

    Grid* population = new Grid[params.population_size];
    int* scores = new int[params.population_size];
    Grid* d_pop;
    int* d_scores;
    Parameters* d_params;

    Grid bs{};
    {
        bs.cases[0][0] = Case{ 0, 1, 3 };
        bs.cases[1][0] = Case{ 2 };
        bs.cases[2][0] = Case{ 0, 0, 4 };
        bs.cases[3][0] = Case{ 0, 2, 5 };
        bs.cases[4][0] = Case{ 0, 1, 5 };
        bs.cases[5][0] = Case{ 0, 4, 5 };
        bs.cases[6][0] = Case{ 0, 2, 5 };


        bs.cases[0][1] = Case{ 0, 1, 4 };
        bs.cases[1][1] = Case{ 0, 1, 4 };
        bs.cases[2][1] = Case{ 0, 1, 4 };
        bs.cases[3][1] = Case{ 1, 0 };
        bs.cases[4][1] = Case{ 1, 0 };
        bs.cases[5][1] = Case{ 0, 2, 6 };
        bs.cases[6][1] = Case{ 0, 1, 7 };

        bs.cases[0][2] = Case{ 0, 1, 4 };
        bs.cases[1][2] = Case{ 0, 2, 3 };
        bs.cases[2][2] = Case{ 1, 0 };
        bs.cases[3][2] = Case{ 1, 0 };
        bs.cases[4][2] = Case{ 1, 0 };
        bs.cases[5][2] = Case{ 0, 1, 7 };
        bs.cases[6][2] = Case{ 0, 1, 7 };

        bs.cases[0][3] = Case{ 0, 1, 3 };
        bs.cases[1][3] = Case{ 1, 0 };
        bs.cases[2][3] = Case{ 1, 0 };
        bs.cases[3][3] = Case{ 1, 0 };
        bs.cases[4][3] = Case{ 1, 0 };
        bs.cases[5][3] = Case{ 0, 1, 7 };
        bs.cases[6][3] = Case{ 0, 0, 0 };

        bs.cases[0][4] = Case{ 0, 0, 5 };
        bs.cases[1][4] = Case{ 2 };
        bs.cases[2][4] = Case{ 1, 0 };
        bs.cases[3][4] = Case{ 1, 0 };
        bs.cases[4][4] = Case{ 1, 0 };
        bs.cases[5][4] = Case{ 1, 0 };
        bs.cases[6][4] = Case{ 0, 1, 7 };

        bs.cases[0][5] = Case{ 0, 4, 2 };
        bs.cases[1][5] = Case{ 1, 0 };
        bs.cases[2][5] = Case{ 1, 1 };
        bs.cases[3][5] = Case{ 0, 1, 1 };
        bs.cases[4][5] = Case{ 0, 1, 1 };
        bs.cases[5][5] = Case{ 1, 0 };
        bs.cases[6][5] = Case{ 0, 1, 0 };

        bs.cases[0][6] = Case{ 0, 3, 2 };
        bs.cases[1][6] = Case{ 1, 0 };
        bs.cases[2][6] = Case{ 1, 0 };
        bs.cases[3][6] = Case{ 0, 1, 7 };
        bs.cases[4][6] = Case{ 0, 1, 7 };
        bs.cases[5][6] = Case{ 0, 4, 0 };
        bs.cases[6][6] = Case{ 0, 1, 0 };

        bs.limits[0] = 4;
        bs.limits[1] = 1;
        bs.limits[2] = 3;
        bs.limits[3] = 1;
    }

    // Prepare
    for(int i = 0 ; i < params.population_size ; ++i)
        copy_tab(population[i], bs);
    //create_population(params.population_size, population);
    cudaMalloc((void**)&d_pop, sizeof(Grid) * params.population_size);
    cudaMalloc((void**)&d_scores, sizeof(int) * params.population_size);
    cudaMalloc((void**)&d_params, sizeof(Parameters));

    cudaMemcpy(d_pop, population, sizeof(Grid) * params.population_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scores, scores, sizeof(int) * params.population_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, &params, sizeof(Parameters), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float smallest_time = 1e9;
    int recomended_threads = 0;

    for (int i = 0; i < 11; ++i) {
        int threads_per_block = pow(2, i);
        int blocks = params.population_size / threads_per_block + 1;
        // Start record
        cudaEventRecord(start, 0);
        // Do something on GPU
        for(int j = 0;j<100;++j)
            cuda_eval <<<blocks, threads_per_block >>> (d_params, d_pop, d_scores);
        // Stop event
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
        if (elapsedTime < smallest_time)
        {
            smallest_time = elapsedTime;
            recomended_threads = pow(2, i);
        }
        if(verbose)
            std::cout << "Threads per block : " << threads_per_block << "\tTotal time : " << elapsedTime 
                << "\tTime per sim:" << elapsedTime / 100 << std::endl;
    }
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if(verbose)
        std::cout << "Recommended thread per block : " << recomended_threads << std::endl;
    return recomended_threads;
}

