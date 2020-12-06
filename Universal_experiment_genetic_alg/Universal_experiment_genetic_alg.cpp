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
#include <mutex>


// Simulation parameters
#define POPULATION_SIZE 200
#define MUTATION_RATE 0.2
#define RETAIN_PERCENT 0.05
#define MAX_ITERATIONS 100000
#define MIN_MUTATIONS 1
#define MAX_MUTATIONS 5

// do not change
#define DIM 7
#define MAX_5T_ARROWS 4
#define MAX_ROT_ARROWS 3
#define MAX_INF_ARROWS 1
#define MAX_REFRESH 1

/* Prototypes */
unsigned short create_object(int indice);
int simulate(int indice, unsigned short individual[POPULATION_SIZE][DIM][DIM]);
void populate(unsigned short population[POPULATION_SIZE][DIM][DIM]);
void repr(unsigned short individual[DIM][DIM]);
void quicksortIndices(int values[], int indices[]);
void quicksortIndices(int values[], int indices[], int low, int high);
void swap(int i, int j, int values[], int indices[]);
void copy_tab(unsigned short p1[DIM][DIM], unsigned short p2[DIM][DIM]);
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


int usage[POPULATION_SIZE][4] = { {0} };
// [arrows_5t, arrows_inf, arrows_rot, refresh_orb]

unsigned short create_object(int indice) {
    unsigned short individual = 0;
    int tmp;

    switch (rand() % 3) {
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
            tmp = (usage[indice][3] < MAX_REFRESH) ? rand() % 3 : rand() % 2;
            usage[indice][3] += (tmp == 0b10);
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
void refresh_individual(unsigned short individual[DIM][DIM]) {
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            individual[i][j] = individual[i][j] & 0b1111100011111000;
        }
    }
}

/**
 * Simulate a game for an individual at indice indice
*/
int simulate(int indice, unsigned short population[POPULATION_SIZE][DIM][DIM])
{
    char movements[8][2] = { {-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0} };
    char x = -1, y = 6;
    char cur_dir = 3;

    bool end = false;
    int score = 0;
    char uses;
    char type;
    char dir;

    unsigned short individual[DIM][DIM];
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
                        individual[x][y] = individual[x][y] & 0b1111100011111111;
                        individual[x][y] |= (uses << 8);
                        cur_dir = dir;
                    }
                    else if (type == 0b011) { // infinite
                        cur_dir = dir;
                    } 
                    else if (type == 0b100) { // rotating
                        cur_dir = dir;
                        dir = (dir + 1) % 8;

                        individual[x][y] = individual[x][y] & 0b1111111100011111;
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
                                //refresh grid()
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
    return score;

}

/**
 * Creates the initial population with random objects 
 */
void populate(unsigned short population[POPULATION_SIZE][DIM][DIM]) {

    for (int i = 0; i < POPULATION_SIZE; ++i) {
        for (int x = 0; x < DIM; ++x) {
            for (int y = 0; y < DIM; ++y) {
                population[i][x][y] = create_object(i);
            }
        }
    }
}

/**
 *  Give a visual representation of the grid in the console
 * WINDOWS DEPENDENT !
 */
void repr(unsigned short individual[DIM][DIM])
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


void quicksortIndices(int values[], int indices[]) {
    //int indices[] = new int[values.length];
    for (int i = 0; i < POPULATION_SIZE; i++)
        indices[i] = i;
    quicksortIndices(values, indices, 0, POPULATION_SIZE - 1);
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
void quicksortIndices(int values[], int indices[], int low, int high) {
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
        quicksortIndices(values, indices, low, h);
    if (high > l)
        quicksortIndices(values, indices, l, high);
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
void breed(int indice, int parent_a, int parent_b, unsigned short population[POPULATION_SIZE][DIM][DIM]) {
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
void mutate_individual(int indice, unsigned short population[POPULATION_SIZE][DIM][DIM]) {
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

    population[indice][x][y] = create_object(indice);
}

/**
 * Copy the content of tab p1 into tab p2
 */
void copy_tab(unsigned short p1[DIM][DIM], unsigned short p2[DIM][DIM]) {
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            p1[i][j] = p2[i][j];
        }
    }
}


int main()
{
    unsigned short population[POPULATION_SIZE][DIM][DIM];
    int scores[POPULATION_SIZE] = { 0 };
    int scores_indices[POPULATION_SIZE];

    float avg_score;
    unsigned short fittest[DIM][DIM];
    int fit_score=0;

    time_t t =time(NULL);
    std::cout << "Seed:" << t << std::endl;
    srand(t);

    //threads
    /*
    std::vector<std::thread> Pool;
    for (int ii = 0; ii < 8; ii++)
    {
        Pool.push_back(std::thread(Infinite_loop_function));
    }*/

    //initial state
    populate(population);

    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
        // evalute the population
        for (int i = 0; i < POPULATION_SIZE; i++) {
            //repr(population[i]);
            scores[i] = simulate(i, population);
        }

        // sort 
        avg_score = 0;

        for (int i = 0; i < POPULATION_SIZE; ++i) {
            scores_indices[i] = i;
            avg_score += scores[i];
        }
        avg_score /= (float)POPULATION_SIZE;
        quicksortIndices(scores, scores_indices);

        //save the fittest and his score
        if (scores[0] < fit_score) {
            copy_tab(fittest, population[scores_indices[0]]);
            fit_score = scores[0];
            repr(fittest);
        }

        for (int i = 0; i < POPULATION_SIZE; ++i) {
            //replace with RETAIN_PERCENT
            if (i > RETAIN_PERCENT * POPULATION_SIZE) {
                int parent_a, parent_b;
                parent_a = scores_indices[0];//rand() % (int)(RETAIN_PERCENT * POPULATION_SIZE)];
                parent_b = scores_indices[0];// rand() % (int)(RETAIN_PERCENT * POPULATION_SIZE)];
                //new individual with NOT random parents BECAUSE ITS ANNOYING
                breed(scores_indices[i], parent_a, parent_b, population);
            }

            // mutate
            int mutation_count = rand() % (MAX_MUTATIONS - MIN_MUTATIONS) + MIN_MUTATIONS;
            for (int j = 0; j < mutation_count; ++j) {
                if (rand() / (float)RAND_MAX < MUTATION_RATE) {
                    mutate_individual(i, population);
                }
            }
        }
        if (iteration % 1000 == 0) {
            std::cout << "Iteration : " << iteration << "\tavg_score : " << avg_score << "\tBest : " << scores[0] << std::endl;
        }


    }
    std::cout << "Best\nFitness : " << scores[0] << std::endl;
    repr(fittest);
    
    return 0;
}
