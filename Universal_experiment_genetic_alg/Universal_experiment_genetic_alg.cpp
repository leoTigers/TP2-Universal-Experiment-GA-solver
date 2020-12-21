/*
* Attempts to solve the universal experiment in "the perfect tower 2"
* Author TigTig#0621
* Date 20 dec 2020
* Version 0.1
*/

#include <iostream>   
#include <time.h> 
#include <vector>
#include "console.h"
#include <thread>

#include "cuda_compute.cuh"
#include "structs.h"

#ifdef max
#undef max
#endif // removes the max macro defined in some ide

// DEPRECATED
// CPU thread management 
#define USE_THREADS 0
#define THREAD_COUNT 5

// GAME CONSTANTS
#define DIM 7

// Simulation parameters, shouldn't change assuming you unlocked everything (3 prestiges)
#define MAX_5T_ARROWS 4
#define MAX_ROT_ARROWS 3
#define MAX_INF_ARROWS 1
#define MAX_REFRESH 1
#define REFLECT_UNLOCK 1


/* End structures */

/* Prototypes */
void create_population(int population_size, Grid* population);
void create_individual(Grid& individual);
void create_object(Case& object, char limits[]);
void evaluate(int population_size, Grid* population, int* scores, int* scores_indices);
Parameters parse_parameters(int argc, char* argv);
void repr(Grid& individual);
void copy_tab(Grid &g1, Grid &g2);
void breed(int population_size, int indice, int parent_a, int parent_b, Grid* population);
void refresh_individual(Grid& individual);
int simulate(int population_size, int indice, Grid* population);
void mutate_individual(int population_size, Grid& individual);

void quicksortIndices(int population_size, int values[], int indices[]);
void quicksortIndices(int population_size, int values[], int indices[], int low, int high);
void swap(int i, int j, int values[], int indices[]);
void wait_on_enter();

void multi_sim(int population_size, int start, int end, Grid* population, int* scores);

/* End prototypes*/


/**
 * Creates the initial population with random objects
 * 
 * @param population_size the population size
 * @param population the population of Grids
 */
void create_population(int population_size, Grid* population) {
    for (int i = 0; i < population_size; ++i) {
        create_individual(population[i]);
    }
}

/**
 * Create an individual
 * @param individual a reference to the individual to create
 */
void create_individual(Grid& individual) {
    // fill base limits uses with zeros
    for (int i = 0; i < 4; ++i)
        individual.limits[i] = 0;
    individual.changed = true;
    individual.score = 0;

    // fill the grid with random objects
    for (int x = 0; x < DIM; ++x) 
        for (int y = 0; y < DIM; ++y) 
            create_object(individual.cases[x][y], individual.limits);
}

/**
 * Create an object
 * @param object a reference to the object to create
 * @param limits the limits for the grid
 */
void create_object(Case& object, char limits[]) {
    char type = rand() % (REFLECT_UNLOCK ? 3 : 2);
    object.type = type;
    switch (type) {
        case Object_type::ARROW: //arrow
            // pick an arrow type within grid object limits
            do {
                object.o_type = rand() % 5;
            } while ((object.o_type == Arrow_type::FIVE_USES && limits[0] == MAX_5T_ARROWS) ||
                (object.o_type == Arrow_type::INFINITE_USES && limits[1] == MAX_INF_ARROWS) ||
                (object.o_type == Arrow_type::ROTATING && limits[2] == MAX_ROT_ARROWS));

            if (object.o_type > 1)
                limits[object.o_type - 2]++;
            object.dir = rand() & 7;
            break;
        case Object_type::ORB: //orb
            // orb can be normal or refresh
            object.o_type = (limits[3] < MAX_REFRESH) ? rand() & 1 : 0;
            limits[3] += (object.o_type & 1);
            break;
    }
}

void evaluate(int population_size, Grid* population, int* scores, int* scores_indices) {
    std::vector<std::thread> Pool(population_size);

    if (!USE_THREADS) {
        for (int i = 0; i < population_size; i++) {
            //repr(population[i]);
            scores[i] = simulate(population_size, i, population);
        }
    }
    else {
        Pool.clear();
        for (int i = 0; i < THREAD_COUNT; i++)
            Pool.push_back(std::thread(multi_sim, population_size, (i * population_size) / THREAD_COUNT, ((i + 1) * population_size) / THREAD_COUNT, population, scores));

        for (int i = 0; i < THREAD_COUNT; i++)
            Pool[i].join();
    }
    /**/

}


Parameters parse_parameters(int argc, char* argv) {
    Parameters params{};
    if (argc < 2)
    {
        std::cout << "Enter the population size:";
        std::cin >> params.population_size;
        if (params.population_size < 1) {
            std::cout << "YOU TRIED" << std::endl;
            wait_on_enter();
            exit(-1);
        }

        std::cout << "Enter the retain rate (the amount of individuals not dying each round, between 0 and 1): ";
        std::cin >> params.retain_rate;
        if (params.retain_rate < 0 || params.retain_rate>1) {
            std::cout << "YOU TRIED" << std::endl;
            wait_on_enter();
            exit(-1);
        }

        std::cout << "Enter the mutation rate (the chance of a mutation occuring, between 0 and 1): ";
        std::cin >> params.mutation_rate;
        if (params.mutation_rate < 0 || params.mutation_rate>1) {
            std::cout << "YOU TRIED" << std::endl;
            wait_on_enter();
            exit(-1);
        }

        std::cout << "Enter the number of iterations: ";
        std::cin >> params.max_iterations;
        if (params.max_iterations < 0) {
            std::cout << "YOU TRIED" << std::endl;
            wait_on_enter();
            exit(-1);
        }

        std::cout << "Enter the minimum amount of mutations that CAN occur (still depends on mutation rate): ";
        std::cin >> params.min_mutations;
        if (params.min_mutations < 0) {
            std::cout << "YOU TRIED" << std::endl;
            wait_on_enter();
            exit(-1);
        }

        std::cout << "Enter the maximum amount of mutations that CAN occur (still depends on mutation rate): ";
        std::cin >> params.max_mutations;
        if (params.max_mutations < 0 || params.max_mutations < params.min_mutations) {
            std::cout << "YOU TRIED" << std::endl;
            wait_on_enter();
            exit(-1);
        }
    }
    return params;
}

/**
 *  Give a visual representation of the grid in the console
 * WINDOWS DEPENDENT !
 */
void repr(Grid &individual)
{
    char arrows_dir[8] = { (char)218, '^', (char)191, '>', (char)217, 'v', (char)192, '<' };
    for (int y = 0; y < DIM; ++y) {
        for (int x = 0; x < DIM; ++x) {
            switch (individual.cases[x][y].type) {
            case Object_type::ARROW: 
                switch (individual.cases[x][y].o_type)
                {
                case Arrow_type::ONE_USE: 
                    console::setColor(15, 0);
                    break;
                case Arrow_type::THREE_USES: 
                    console::setColor(6, 0);
                    break;
                case Arrow_type::FIVE_USES:
                    console::setColor(11, 0);
                    break;
                case Arrow_type::INFINITE_USES: 
                    console::setColor(0, 15);
                    break;
                case Arrow_type::ROTATING: 
                    console::setColor(13, 7);
                    break;
                }
                std::cout << arrows_dir[individual.cases[x][y].dir];
                console::setColor(15, 0);
                break;
            case Object_type::ORB: 
                switch (individual.cases[x][y].o_type)
                {
                case Orb_type::NORMAL:
                    console::setColor(12, 0);
                    break;
                case Orb_type::REFRESH:
                    console::setColor(5, 0);
                    break;
                }
                std::cout << "O";
                console::setColor(15, 0);
                break;
            case Object_type::REFLECT:
                std::cout << "R";
                break;
            }
            std::cout << " ";
        }
        std::cout << std::endl << std::endl;
    }
}

/**
 * Copy the content of Grid g2 into Grid p1
 * @param g1 the receiving Grid
 * @param g2 the source Grid
 */
void copy_tab(Grid &g1, Grid &g2) {
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            g1.cases[i][j] = g2.cases[i][j];
        }
    }
    g1.changed = g2.changed;
    for (int i = 0; i < 4; ++i)
        g1.limits[i] = g2.limits[i];
    g1.score = g2.score;
}

/**
 * Breed a new individual with 2 parents, update object limits
 * 
*/
void breed(int population_size, int indice, int parent_a, int parent_b, Grid* population) {
    // Reset limits for new individual
    for (int i = 0; i < 4; i++)
        population[indice].limits[i] = 0;

    int crosspoint = rand() % 49; // the case at which ponit we'll take values from parent_b instead of parent_a
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            Case tmp;
            if (i * DIM + j < crosspoint) { // Normal addition
                tmp = population[parent_a].cases[i][j];
                switch (tmp.type) {
                case Object_type::ARROW: 
                    if (tmp.o_type > 1)
                        population[indice].limits[tmp.o_type - 2]++;
                    break;
                case Object_type::ORB: 
                    if (tmp.o_type == 1)
                        population[indice].limits[3]++;
                    break;
                }
            }
            else { // Addition with limit check
                tmp = population[parent_b].cases[i][j];
                boolean valid_insertion = true;
                switch (tmp.type) {
                case Object_type::ARROW: 
                    switch (tmp.o_type)
                    {
                        case Arrow_type::FIVE_USES:
                            if (population[indice].limits[0] >= MAX_5T_ARROWS)
                                valid_insertion = false;
                            break;
                        case Arrow_type::INFINITE_USES:
                            if (population[indice].limits[1] >= MAX_INF_ARROWS)
                                valid_insertion = false;
                            break;
                        case Arrow_type::ROTATING:
                            if (population[indice].limits[2] >= MAX_ROT_ARROWS)
                                valid_insertion = false;
                            break;
                        default:
                            break;
                    }
                    if (!valid_insertion)
                        create_object(tmp, population[indice].limits);
                    else {
                        if (tmp.o_type > 1)
                            population[indice].limits[tmp.o_type - 2]++;
                    }
                    break;
                case Object_type::ORB: 
                    if (tmp.o_type == Orb_type::REFRESH) {
                        if (population[indice].limits[3] >= MAX_REFRESH)
                            create_object(tmp, population[indice].limits);
                        else {
                            population[indice].limits[3]++;
                        }
                    }
                    break;
                }
            }
            population[indice].cases[i][j] = tmp;
        }
    }
    population[indice].changed = true;
}

/**
 * set arrows/orb/reflect usage count to 0
*/
void refresh_individual(Grid &individual) {
    for (int i = 0; i < DIM; ++i)
        for (int j = 0; j < DIM; ++j) {
            individual.cases[i][j].current_uses = 0;
            individual.cases[i][j].c_dir = individual.cases[i][j].dir;
        }
            
}

/**
 * Simulate a game for an individual at indice indice
*/
int simulate(int population_size, int indice, Grid* population)
{
    //if grid hasen't changed
    if (!population[indice].changed)
        return population[indice].score;

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
    refresh_individual(population[indice]);

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
                                refresh_individual(population[indice]);
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

    return score;
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


/**
 * Mutate an individual and manages objet limits
*/
void mutate_individual(int population_size, Grid &individual) {
    // pick a random position
    int x = rand() % DIM;
    int y = rand() % DIM;

    int type;

    // update limits
    switch (individual.cases[x][y].type) {
    case Object_type::ARROW: 
        type = individual.cases[x][y].o_type;
        switch (type) {
            case Arrow_type::FIVE_USES:
                individual.limits[0]--;
                break;
            case Arrow_type::INFINITE_USES:
                individual.limits[1]--;
                break;
            case Arrow_type::ROTATING:
                individual.limits[2]--;
                break;
        }
        break;
    case Object_type::ORB:
        if(individual.cases[x][y].o_type == Orb_type::REFRESH)
            individual.limits[3]--;
        break;
    }
    
    // create new object
    create_object(individual.cases[x][y], individual.limits);
    individual.changed = true;
}

void status(int population_size, Grid* population, int iteration, int max_iter, Grid &fittest, int* scores, time_t start, time_t now) {
    float avg_score = 0;
    for (int i = 0; i < population_size; ++i)
        avg_score += population[i].score;
    avg_score /= (float)population_size;
    system("cls");
    console::gotoxy(0, 0);
    std::cout << "Iteration : " << iteration << "/" << max_iter;
    console::gotoxy(30, 0);
    std::cout << "Avg_score : " << -avg_score;
    console::gotoxy(53, 0);
    std::cout << "Current best : " << -scores[0];
    console::gotoxy(73, 0);
    std::cout << "Elapsed time : " << (now - start) << "s";
    console::gotoxy(97, 0);
    if (now != start)
        std::cout << (iteration / (now - start)) << " iter/s";
    else
        std::cout << "0 iter/s";
    console::gotoxy(0, 1);
    std::cout << "Ever best : " << -fittest.score;
    console::gotoxy(0, 2);
    repr(fittest);
}

//evaluate(params.population_size, population, scores, scores_indices);
 
void multi_sim(int population_size, int start, int end, Grid *population, int *scores) {
    for(int i = start;i<end;++i){
        scores[i] = simulate(population_size, i, population);
    }
}

/**
 * @brief Pause until user input
 */
void wait_on_enter()
{
    std::string dummy; 
    std::cin.clear(); 
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cout << "Enter to continue..." << std::endl;
    std::getline(std::cin, dummy);
}

int main(int argc, char* argv)
{
    std::cout << "This program is made with little research and no prior knowledge on this subject. Please manage your expectations." << std::endl; 
    std::cout << "Please don't try to break something, it WILL break." << std::endl;
    std::cout << "Source code available at https://github.com/leoTigers/TP2-Universal-Experiment-GA-solver" << std::endl;
    std::cout << "Feel free to experiment and maybe improve this program / fix the mistakes I made :D" << std::endl;
    std::cout << "This program can run with parameters, to learn more about symbols and parameters, run with -h" << std::endl;
    


    // Simulation parameters
    /*
    int population_size;
    int max_iterations;
    int min_mutations;
    int max_mutations;
    int min_lifetime;
    float retain_rate;
    float mutation_rate;
    */
    Parameters params{};// {100, 50000, 1, 6, .2, .3};//= parse_parameters(argc, argv);
    params.population_size = 10000;
    params.max_iterations = 200000;
    params.min_mutations = 0;
    params.max_mutations = 6;
    params.retain_rate = 0.8f;
    params.mutation_rate = .1f;

    Grid *population = new Grid[params.population_size];
    Grid fittest{};

    int *scores = new int[params.population_size];
    int *scores_indices = new int[params.population_size];
    //float avg_score; 

    time_t t =time(NULL);
    time_t start = time(NULL);
    time_t dt;
    std::cout << "Seed:" << start << std::endl;
    srand(start);

    std::cout << sizeof(Grid) * params.population_size<< "o"<<std::endl;
    //wait_on_enter();

    //initial state
    create_population(params.population_size, population);

    for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
        // evalute the population
        //evaluate(params.population_size, population, scores, scores_indices);

        ce(params.population_size, population);

        // sort 
        //avg_score = 0;
        
        for (int i = 0; i < params.population_size; ++i) {
            scores_indices[i] = i;
            scores[i] = population[i].score;
        //    avg_score += scores[i];
        }
        //avg_score /= (float)params.population_size;
        quicksortIndices(params.population_size, scores, scores_indices);

        //save the fittest and his score
        if (scores[0] < fittest.score) {
            copy_tab(fittest, population[scores_indices[0]]);
            //status(params.population_size, population, iteration, params.max_iterations, fittest, scores, start, time(NULL));
            //std::cout << "Score:" << -fittest.score << std::endl;
            //repr(fittest);
        }

        for (int i = 0; i < params.population_size; ++i) {
            //replace with RETAIN_PERCENT
            if (i > params.retain_rate * params.population_size) {
                int parent_a, parent_b;
                parent_a = scores_indices[rand() % (int)(params.retain_rate * params.population_size)];
                parent_b = scores_indices[rand() % (int)(params.retain_rate * params.population_size)];
                breed(params.population_size, scores_indices[i], parent_a, parent_b, population);
            }
            // mutate
            // TODO : REWORK

            // forced mutations
            for (int j = 0; j < params.min_mutations; ++j) {
                mutate_individual(params.population_size, population[i]);
                population[i].changed = true;
            }
            for (int j = 0; j < params.max_mutations - params.min_mutations; ++j) {
                if (rand() / (float)RAND_MAX < params.mutation_rate) {
                    mutate_individual(params.population_size, population[i]);
                    population[i].changed = true;
                }
            }
        }
        // TODO : REWORK TIME DEPENDENT
        dt = time(NULL);
        if (dt-t > 2) {
            status(params.population_size, population, iteration, params.max_iterations, fittest, scores, start, t);
            t = dt;
            /*if (changed) {
                //repr(fittest);
                changed = false;
            }*/
        }


    }

    std::cout << "Best fitness : " << -fittest.score << std::endl;
    repr(fittest);

    std::cout << "Seed:" << start << std::endl;
    wait_on_enter();

    return 0;
}
