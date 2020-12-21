#include "cuda_compute.cuh"

__global__ void cuda_eval(int* population_size, Grid* population) {
    
    int indice = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (indice >= *population_size)
        return;
    //if grid hasen't changed
    if (!population[indice].changed)
        return;

    char movements[8][2] = { {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0} };
    // starting pos
    char x = -1, y = 6;
    char cur_dir = 3;

    bool end = false;
    int score = 0;
    char uses;
    char type;
    char dir;

    //copy_tab(individual, population[indice]);
    for (int i = 0; i < DIM; ++i){
        for (int j = 0; j < DIM; ++j) {
            population[indice].cases[i][j].current_uses = 0;
            population[indice].cases[i][j].c_dir = population[indice].cases[i][j].dir;
        }
    }

    do {
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
                                population[indice].cases[i][j].c_dir = population[indice].cases[i][j].dir;
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
}

void ce(int population_size, Grid* population) {

    Grid* d_population;
    //cudaMallocManaged(&res, sizeof(Grid));
    int* d_population_size;

    cudaMalloc((void**)&d_population, population_size * sizeof(Grid));
    cudaMalloc((void**)&d_population_size, sizeof(int));

    cudaMemcpy(d_population, population, sizeof(Grid) * population_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_population_size, &population_size, sizeof(int), cudaMemcpyHostToDevice);

    cuda_eval<<<100, 256>>>(d_population_size, d_population);
    cudaDeviceSynchronize();
    cudaMemcpy(population, d_population, sizeof(Grid) * population_size, cudaMemcpyDeviceToHost);

    cudaFree(d_population);
    cudaFree(d_population_size);
}

