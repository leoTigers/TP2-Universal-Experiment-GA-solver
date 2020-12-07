/*
* Solves the universal experiment in "the perfect tower 2"
* Author TigTig#0621
* Date 06 dec 2020
* Version 1.0
*/

#include <iostream>   
#include <time.h> 
#include <cstdlib>
#include <vector>
#include "console.h"
#include <thread>
#include <limits>

#undef max

// Simulation parameters
/*
#define POPULATION_SIZE 400
#define MUTATION_RATE 0.2
#define RETAIN_PERCENT 0.2
#define MAX_ITERATIONS 200000
#define MIN_MUTATIONS 0
#define MAX_MUTATIONS 5
*/

#define USE_THREADS 1
#define THREAD_COUNT 20


// do not change
#define DIM 7
#define MAX_5T_ARROWS 4
//#define MAX_5T_ARROWS 0
#define MAX_ROT_ARROWS 3
#define MAX_INF_ARROWS 1
#define MAX_REFRESH 1
#define REFLECT_UNLOCK 1

/* Prototypes */
unsigned short create_object(int indice, int **usage);
int simulate(int population_size, int indice, unsigned short ***population, int **usage);
void populate(int population_size, unsigned short ***population, int **usage);
void repr(unsigned short **individual);
void quicksortIndices(int population_size, int values[], int indices[]);
void quicksortIndices(int population_size, int values[], int indices[], int low, int high);
void swap(int i, int j, int values[], int indices[]);
void copy_tab(unsigned short **p1, unsigned short **p2);
/* End prototypes*/


/// OBJECT REPR
// object type [T]
/*
Arrow
Orb
Reflect
3 types => 2bits
*/

// Arrows
/*
type (1:000, 3:001, 5:010, infinite:011, rotating:100) 3bits [A]
uses : 1, 3, 5, infinite (3 bits) [U]
direction (8 possibles => 3bits) [D] 
    STARTING TOPLEFT + CLOCKWISE
*/

// Orbs
/*
type (0:red 1:blue, 2:purple ) 2bits [t]
used (for purple) 1bit [b]
*/

//Reflect
/*
uses (2times => 2bits) [u]
*/

// combine as short
// TTAAAUUU_DDDttbuu    


//int usage[POPULATION_SIZE][4] = { {0} };
// [arrows_5t, arrows_inf, arrows_rot, refresh_orb]

unsigned short create_object(int indice, int **usage) {
    unsigned short individual = 0;
    int tmp;

    switch (rand() % (REFLECT_UNLOCK?3:2)) {
        case 0: //arrow
            individual |= (0b01 << 14);
            do {
                tmp = rand() % 5;
            } while ((tmp == 0b10 && usage[indice][0] == MAX_5T_ARROWS) ||
                (tmp == 0b11 && usage[indice][1] == MAX_INF_ARROWS) ||
                (tmp == 0b100 && usage[indice][2] == MAX_ROT_ARROWS));
            switch (tmp) {
                case 0b10:
                    usage[indice][0]++;
                    break;
                case 0b11:
                    usage[indice][1]++;
                    break;
                case 0b100:
                    usage[indice][2]++;
                    break;
                default:
                    break;
            }
            individual |= (tmp << 11);
            individual |= ((rand() % 8) << 5);
            break;
        case 1: //orb
            individual |= (0b10 << 14);
            // we don't need the red ord
            tmp = 1 + ((usage[indice][3] < MAX_REFRESH) ? rand() % 2 : rand() % 1);
            usage[indice][3] += (tmp & 0b10);
            
            individual |= (tmp << 3);
            break;
        case 2: //reflect
            individual |= (0b11 << 14);
            break;
    }
    return individual;
}

/**
 * set arrows/orb/reflect usage count to 0
*/
void refresh_individual(unsigned short **individual) {
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            individual[i][j] &= 0b1111100011111000;
        }
    }
}

/**
 * Simulate a game for an individual at indice indice
*/
int simulate(int population_size, int indice, unsigned short ***population, int **usage)
{
    char movements[8][2] = { {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0} };
    char x = -1, y = 6;
    char cur_dir = 3;

    bool end = false;
    int score = 0;
    char uses;
    char type;
    char dir;

    unsigned short** individual = new unsigned short* [DIM];
    for (int i = 0; i < DIM; i++)
        individual[i] = new unsigned short[DIM];
    copy_tab(individual, population[indice]);

    refresh_individual(individual);

    do {
        //move
        x += movements[cur_dir][0];
        y += movements[cur_dir][1];

        //check exit cond
        if (x < 0 || x >= DIM || y < 0 || y >= DIM) {
            end = true;
        }
        else {
            switch ((individual[x][y]>>14)&0b11) {
                case 0b01: //arrow
                    type = (individual[x][y] >> 11) & 0b111;
                    uses = (individual[x][y] >> 8) & 0b111;
                    dir = (individual[x][y] >> 5) & 0b111;
                    if ((type == 0b000 && uses < 1) || (type == 0b001 && uses < 3) || (type == 0b010 && uses < 5)) {
                        uses++;
                        individual[x][y] &= 0b1111100011111111;
                        individual[x][y] |= (uses << 8);
                        cur_dir = dir;
                    }
                    else if (type == 0b011) { // infinite
                        cur_dir = dir;
                    } 
                    else if (type == 0b100) { // rotating
                        cur_dir = dir;
                        dir = (dir + 1) % 8;

                        individual[x][y] &= 0b1111111100011111;
                        individual[x][y] |= (dir << 5);
                    }

                    break;
                case 0b10: //orb
                    switch ((individual[x][y] >> 3) & 0b11)
                    {
                        case 0b00: //red
                            score++;
                            break;
                        case 0b01: //blue
                            score--;
                            break;
                        case 0b10:
                            if (!(individual[x][y] & 0b100)) { //refresh not used
                                refresh_individual(individual);
                                individual[x][y] |= 0b100;
                            }
                            break;
                    }
                    break;
                case 0b11: //reflect
                    uses = (individual[x][y]) & 0b11;
                    if (uses < 2) {
                        uses++;
                        if (cur_dir & 1)
                            cur_dir = (cur_dir + 4) % 8;
                        else if (cur_dir == 0)
                            cur_dir = 6;
                        else
                            cur_dir -= 2;

                        individual[x][y] &= 0b1111111111111100;
                        individual[x][y] |= uses;
                    }

                    break;
            }
        }
    } while (!end);

    // deallocate memory
    for (int i = 0; i < DIM; i++)
        delete[] individual[i];
    delete[] individual;


    return score;

}

/**
 * Creates the initial population with random objects 
 */
void populate(int population_size, unsigned short ***population, int **usage) {

    for (int i = 0; i < population_size; ++i) {
        for (int x = 0; x < DIM; ++x) {
            for (int y = 0; y < DIM; ++y) {
                population[i][x][y] = create_object(i, usage);
            }
        }
    }
}

/**
 *  Give a visual representation of the grid in the console
 * WINDOWS DEPENDENT !
 */
void repr(unsigned short **individual)
{
    char arrows_dir[8] = { (char)218, '^', (char)191, '>', (char)217, 'v', (char)192, '<' };
    for (int y = 0; y < DIM; ++y) {
        for (int x = 0; x < DIM; ++x) {
            switch (individual[x][y]>>14) {
                case 0b01: //arrow
                    switch ((individual[x][y] >> 11) & 0b111)
                    {
                    case 0b000: //1
                        console::setColor(15, 0);
                        break;
                    case 0b001: //3
                        console::setColor(6, 0);
                        break;
                    case 0b010: //5
                        console::setColor(11, 0);
                        break;
                    case 0b011: //inf
                        console::setColor(0, 15);
                        break;
                    case 0b100: //rotate
                        console::setColor(13, 7);
                        break;
                    }

                    std::cout << arrows_dir[(individual[x][y]>>5)&0b111];
                    console::setColor(15, 0);
                    break;
                case 0b10: //orb
                    switch ((individual[x][y]>>3)&0b11)
                    {
                    case 0b00:
                        console::setColor(12, 0);
                        break;
                    case 0b01:
                        console::setColor(11, 0);
                        break;
                    case 0b10:
                        console::setColor(5, 0);
                        break;
                    }
                    std::cout << "O";
                    console::setColor(15, 0);
                    break;
                case 0b11:
                    std::cout << "R";
                    break;
            }
            std::cout << " ";
        }
        std::cout << std::endl << std::endl;
    }
}


void quicksortIndices(int population_size, int values[], int indices[]) {
    //int indices[] = new int[values.length];
    for (int i = 0; i < population_size; i++)
        indices[i] = i;
    quicksortIndices(population_size, values, indices, 0, population_size - 1);
    //return indices;
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

/**
 * Breed a new individual with 2 parents, update object limits
 * !! BE CAREFUL, IT DOES NOT CHECK LIMITS WHEN FUSING PARENTS !!
*/
void breed(int population_size, int indice, int parent_a, int parent_b, unsigned short ***population, int **usage) {
    for (int i = 0; i < 4; i++)
        usage[indice][i] = 0;
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            if (rand() & 1)
                population[indice][i][j] = population[parent_a][i][j];
            else
                population[indice][i][j] = population[parent_b][i][j];
            
            int type;

            switch (population[indice][i][j] >> 14) {
            case 0b01: //arrow
                type = (population[indice][i][j] >> 11) & 0b111;
                switch (type) {
                case 0b010:
                    usage[indice][0]++;
                    break;
                case 0b011:
                    usage[indice][1]++;
                    break;
                case 0b100:
                    usage[indice][2]++;
                    break;
                }
                break;
            case 0b10: //orb
                if (((population[indice][i][j] >> 3) & 0b11) == 0b10)
                    usage[indice][3]++;
                break;
            }
        }
    }
}

/**
 * Mutate an individual and manages objet limits
*/
void mutate_individual(int population_size, int indice, unsigned short ***population, int **usage) {
    int x = rand() % DIM;
    int y = rand() % DIM;
    int type;

    switch (population[indice][x][y] >> 14) {
    case 0b01: //arrow
        type = (population[indice][x][y] >> 11) & 0b111;
        switch (type) {
            case 0b010:
                usage[indice][0]--;
                break;
            case 0b011:
                usage[indice][1]--;
                break;
            case 0b100:
                usage[indice][2]--;
                break;
        }
        break;
    case 0b10: //orb
        if(((population[indice][x][y] >> 3) & 0b11)==0b10)
            usage[indice][3]--;
        break;
    }

    population[indice][x][y] = create_object(indice, usage);
}

/**
 * Copy the content of tab p1 into tab p2
 */
void copy_tab(unsigned short **p1, unsigned short **p2) {
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            p1[i][j] = p2[i][j];
        }
    }
}

void multi_sim(int population_size, int start, int end, unsigned short ***population, int *scores, int **usage) {
    for(int i = start;i<end;++i){
        scores[i] = simulate(population_size, i, population, usage);
    }
}

void wait_on_enter()
{
    std::string dummy; 
    std::cin.clear(); 
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cout << "Enter to continue..." << std::endl;
    std::getline(std::cin, dummy);
}

int main()
{
    std::cout << "This program is very bad and won't check if you're entering bad values." << std::endl;
    std::cout << "Therefore please don't try to break something, it WILL break." << std::endl;
    std::cout << "Source code available at https://github.com/leoTigers/TP2-Universal-Experiment-GA-solver" << std::endl;

    int population_size = 50;
    float retain_rate = 0.2f;
    float mutation_rate = 0.2f;
    int max_iterations = 100000;
    int min_mutations = 0; 
    int max_mutations = 5;

    char a;

    std::cout << "Enter the population size:";
    std::cin >> population_size;
    if (population_size < 1) {
        std::cout << "YOU TRIED" << std::endl;
        wait_on_enter();
        return -1;
    }

    std::cout << "Enter the retain rate (the amount of individuals not dying each round, between 0 and 1): ";
    std::cin >> retain_rate;
    if (retain_rate < 0 || retain_rate>1) {
        std::cout << "YOU TRIED" << std::endl;
        wait_on_enter();
        return -1;
    }

    std::cout << "Enter the mutation rate (the chance of a mutation occuring, between 0 and 1): ";
    std::cin >> mutation_rate;
    if (mutation_rate < 0 || mutation_rate>1) {
        std::cout << "YOU TRIED" << std::endl;
        wait_on_enter();
        return -1;
    }

    std::cout << "Enter the number of iterations: ";
    std::cin >> max_iterations;
    if (max_iterations < 0) {
        std::cout << "YOU TRIED" << std::endl;
        wait_on_enter();
        return -1;
    }


    std::cout << "Enter the minimum amount of mutations that CAN occur (still depends on mutation rate): ";
    std::cin >> min_mutations;
    if (min_mutations < 0) {
        std::cout << "YOU TRIED" << std::endl;
        wait_on_enter();
        return -1;
    }

    std::cout << "Enter the maximum amount of mutations that CAN occur (still depends on mutation rate): ";
    std::cin >> max_mutations;
    if (max_mutations < 0 || max_mutations < min_mutations) {
        std::cout << "YOU TRIED" << std::endl;
        wait_on_enter();
        return -1;
    }
    
    unsigned short ***population = new unsigned short**[population_size];
    for (int i = 0; i < population_size; i++)
    {
        population[i] = new unsigned short* [DIM];
        for (int j = 0; j < DIM; j++)
            population[i][j] = new unsigned short[DIM];
    }

    int *scores = new int[population_size];
    int *scores_indices = new int[population_size];

    int **usage = new int*[population_size];
    for (int i = 0; i < population_size; ++i) {
        usage[i] = new int[4];
        for (int j = 0; j < 4; ++j)
            usage[i][j] = 0;
    }

    float avg_score;
    unsigned short** fittest = new unsigned short* [DIM];
    for (int i = 0; i < DIM; ++i) 
        fittest[i] = new unsigned short[DIM];
    
    int fit_score=0;
    bool changed = false;

//Seed:1607275422
    time_t t =time(NULL);
    std::cout << "Seed:" << t << std::endl;
    srand(t)   ;

    //threads
    
    std::vector<std::thread> Pool;
    

    //initial state
    populate(population_size, population, usage);

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        
        // evalute the population
        if (!USE_THREADS) {
            for (int i = 0; i < population_size; i++) {
                scores[i] = simulate(population_size, i, population, usage);
            }
        }
        else {
            Pool.clear();
            for (int i = 0; i < THREAD_COUNT; i++)
                Pool.push_back(std::thread(multi_sim, population_size, (i * population_size) / THREAD_COUNT, ((i + 1) * population_size) / THREAD_COUNT, population, scores, usage));

            for (int i = 0; i < THREAD_COUNT; i++)
                Pool[i].join();
        }
        /**/
        
        
        

        // sort 
        avg_score = 0;

        for (int i = 0; i < population_size; ++i) {
            scores_indices[i] = i;
            avg_score += scores[i];
        }
        avg_score /= (float)population_size;
        quicksortIndices(population_size, scores, scores_indices);

        //save the fittest and his score
        if (scores[0] < fit_score) {
            copy_tab(fittest, population[scores_indices[0]]);
            fit_score = scores[0];
            changed = true;
            std::cout << "Score:" << fit_score << std::endl;
            repr(fittest);
        }

        for (int i = 0; i < population_size; ++i) {
            //replace with RETAIN_PERCENT
            if (i > retain_rate * population_size) {
                int parent_a, parent_b;
                parent_a = scores_indices[0];//rand() % (int)(RETAIN_PERCENT * POPULATION_SIZE)];
                parent_b = scores_indices[0];// rand() % (int)(RETAIN_PERCENT * POPULATION_SIZE)];
                //new individual with NOT random parents BECAUSE ITS ANNOYING
                
                breed(population_size, scores_indices[i], parent_a, parent_b, population, usage);
            }

            // mutate
            int mutation_count = rand() % (max_mutations - min_mutations) + min_mutations;
            for (int j = 0; j < mutation_count; ++j) {
                if (rand() / (float)RAND_MAX < mutation_rate) {
                    mutate_individual(population_size, i, population, usage);
                }
            }
        }
        if (iteration % 1000 == 0) {
            std::cout << "Iteration : " << iteration << "\tavg_score : " << avg_score << "\tBest : " << scores[0] << "\tEver best : " << fit_score << std::endl;
            /*if (changed) {
                //repr(fittest);
                changed = false;
            }*/
        }


    }
    std::cout << "Best fitness : " << fit_score << std::endl;
    repr(fittest);

    std::cout << "Seed:" << t << std::endl;
    wait_on_enter();

    return 0;
}
