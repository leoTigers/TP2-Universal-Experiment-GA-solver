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
#include <random>

#include "cuda_compute.cuh"
#include "structs.h"

#ifdef max
#undef max
#endif // removes the max macro defined in some ide

/* Prototypes */
void create_population(int population_size, Grid* population);
void create_individual(Grid& individual);
void create_object(Case& object, char limits[]);
Parameters parse_parameters(int argc, char* argv[]);
void repr(Grid& individual);
void copy_tab(Grid &g1, Grid &g2);

void wait_on_enter();
/* End prototypes*/


int stoi(const std::string& str, int* p_value, std::size_t* pos = nullptr, int base = 10) {
    // wrapping std::stoi because it may throw an exception
    try {
        *p_value = std::stoi(str, pos, base);
        return 0;
    }
    catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << std::endl;
        return -1;
    }
    catch (const std::out_of_range& oor) {
        std::cerr << "Out of Range error: " << oor.what() << std::endl;
        return -2;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Undefined error: " << e.what() << std::endl;
        return -3;
    }
}

int stof(const std::string& str, float* p_value, std::size_t* pos = nullptr, int base = 10) {
    // wrapping std::stof because it may throw an exception
    try {
        *p_value = std::stof(str, pos);
        return 0;
    }
    catch (const std::invalid_argument& ia) {
        std::cerr << "Invalid argument: " << ia.what() << std::endl;
        return -1;
    }
    catch (const std::out_of_range& oor) {
        std::cerr << "Out of Range error: " << oor.what() << std::endl;
        return -2;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Undefined error: " << e.what() << std::endl;
        return -3;
    }
}

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
    individual.min_lifetime = 20;

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
    object.dir = 0;
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

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)>\n"
        << "Options:\n"
        << "\t-h,--help\t\tShow this help message\n"
        << "\t-p,--population POPULATION_SIZE\tSpecify the size of the population\n"
        << "\t-i,--iterations ITERATION_COUNT\tSpecify the number of iterations\n"
        << "\t-m,--min-mutation MIN_MUTATIONS\tSpecify the minimum of mutations that WILL occur\n"
        << "\t-M,--max-mutation MAX_MUTATIONS\tSpecify the maxiimum of mutations that CAN occur\n"
        << "\t-rr,--retain RETAIN_RATE\tSpecify the retain rate (0.2 means top 20% will survive at the end of a round)\n"
        << "\t-mr,--mutation MUTATION_RATE\tSpecify the mutation rate (between 0 and 1)\n"
        << "\n\n"
        << "Symbols:\n"
        << "\t"<<(char)218 << " ^ " << (char)191 << " > " << (char)217 << " v " << (char)192 << " <" <<" : one use arrows\n";
    console::setColor(6, 0);
    std::cout << "\t" << (char)218 << " ^ " << (char)191 << " > " << (char)217 << " v " << (char)192 << " <";
    console::setColor(15, 0);
    std::cout <<" : three uses arrows\n";
    console::setColor(11, 0);
    std::cout << "\t" << (char)218 << " ^ " << (char)191 << " > " << (char)217 << " v " << (char)192 << " <";
    console::setColor(15, 0);
    std::cout << " : five uses arrows\n";
    console::setColor(0, 15);
    std::cout << "\t" << (char)218 << " ^ " << (char)191 << " > " << (char)217 << " v " << (char)192 << " <";
    console::setColor(15, 0);
    std::cout << " : infinite uses arrows\n";
    console::setColor(13, 7);
    std::cout << "\t" << (char)218 << " ^ " << (char)191 << " > " << (char)217 << " v " << (char)192 << " <";
    console::setColor(15, 0);
    std::cout << " : rotating arrows\n";

    console::setColor(12, 0);
    std::cout << "\t" << "O";
    console::setColor(15, 0);
    std::cout << " : normal orb\n";
    console::setColor(5, 0);
    std::cout << "\t" << "O";
    console::setColor(15, 0);
    std::cout << " : refresh orb\n";

    std::cout << "\tR : reflect" << std::endl;
}

Parameters parse_parameters(int argc, char* argv[]) {
    Parameters params{-1, -1, -1, -1, -1, -1};
    boolean benchmark_required = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            show_usage(argv[0]);
            wait_on_enter();
            exit(1);
        }
        else if (arg == "-p" || arg == "--population") {
            if (i + 1 < argc) {
                stoi(argv[++i], &params.population_size);
            }
            else {
                std::cerr << "--population option requires one argument." << std::endl;
                wait_on_enter();
                exit(1);
            }
        }
        else if (arg == "-i" || arg == "--iterations") {
            if (i + 1 < argc) {
                stoi(argv[++i], &params.max_iterations);
            }
            else {
                std::cerr << "--iterations option requires one argument." << std::endl;
                wait_on_enter();
                exit(1);
            }
        }
        else if (arg == "-m" || arg == "--min-mutation") {
            if (i + 1 < argc) {
                stoi(argv[++i], &params.min_mutations);
            }
            else {
                std::cerr << "--min-mutation option requires one argument." << std::endl;
                wait_on_enter();
                exit(1);
            }
        }
        else if (arg == "-M" || arg == "--max-mutation") {
            if (i + 1 < argc) {
                stoi(argv[++i], &params.max_mutations);
            }
            else {
                std::cerr << "--max-mutation option requires one argument." << std::endl;
                wait_on_enter();
                exit(1);
            }
        }
        else if (arg == "-rr" || arg == "--retain") {
            if (i + 1 < argc) {
                stof(argv[++i], &params.retain_rate);
            }
            else {
                std::cerr << "--retain option requires one argument." << std::endl;
                wait_on_enter();
                exit(1);
            }
        }
        else if (arg == "-mr" || arg == "--mutation") {
            if (i + 1 < argc) {
                stof(argv[++i], &params.mutation_rate);
            }
            else {
                std::cerr << "--mutation option requires one argument." << std::endl;
                wait_on_enter();
                exit(1);
            }
        }
        else if (arg == "--benchmark") {
            benchmark_required = true;
        }
        else if (arg == "--threads") {
            if (i + 1 < argc) {
                stoi(argv[++i], &params.thread_per_block);
            }
            else {
                std::cerr << "--threads option requires one argument." << std::endl;
                wait_on_enter();
                exit(1);
            }
        }
    }

    if (params.population_size < 0) {
        std::cout << "Enter the population size:";
        std::cin >> params.population_size;
        if (params.population_size < 1) {
            std::cout << "Size must be greater than 0" << std::endl;
            wait_on_enter();
            exit(-1);
        }
    }
    if (params.retain_rate < 0) {
        std::cout << "Enter the retain rate (the amount of individuals not dying each round, between 0 and 1): ";
        std::cin >> params.retain_rate;
        if (params.retain_rate < 0 || params.retain_rate>1) {
            std::cout << "Retain rate must be between 0 and 1" << std::endl;
            wait_on_enter();
            exit(-1);
        }
    }
    if (params.mutation_rate < 0) {
        std::cout << "Enter the mutation rate (the chance of a mutation occuring, between 0 and 1): ";
        std::cin >> params.mutation_rate;
        if (params.mutation_rate < 0 || params.mutation_rate>1) {
            std::cout << "Mutation rate must be between 0 and 1" << std::endl;
            wait_on_enter();
            exit(-1);
        }
    }
    if (params.max_iterations < 0) {
        std::cout << "Enter the number of iterations: ";
        std::cin >> params.max_iterations;
        if (params.max_iterations < 0) {
            std::cout << "Iteration count must be greater than 0" << std::endl;
            wait_on_enter();
            exit(-1);
        }
    }
    if (params.min_mutations < 0) {
        std::cout << "Enter the minimum amount of mutations that WILL occur : ";
        std::cin >> params.min_mutations;
        if (params.min_mutations < 0) {
            std::cout << "Minimum mutation can't be negative" << std::endl;
            wait_on_enter();
            exit(-1);
        }
    }
    if (params.max_mutations < 0) {
        std::cout << "Enter the maximum amount of mutations that CAN occur (depends on mutation rate): ";
        std::cin >> params.max_mutations;
        if (params.max_mutations < 0 || params.max_mutations < params.min_mutations) {
            std::cout << "Maximum mutation can't be negative nor smaller than minimum mutation" << std::endl;
            wait_on_enter();
            exit(-1);
        }
    }
    if (benchmark_required)
    {
        benchmark(params, 1);
        wait_on_enter();
        exit(0);
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


void status(int population_size, Grid* population, 
    int iteration, int max_iter, Grid &fittest, 
    int* scores, time_t start, time_t now) {
    float avg_score = 0;
    for (int i = 0; i < population_size; ++i)
        avg_score += scores[i];
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


/**
 * @brief Pause until user input
 */
void wait_on_enter()
{
    std::string dummy; 
    std::cout << "Enter to continue..." << std::endl;
    std::getline(std::cin, dummy);
}

int main(int argc, char* argv[])
{
    system("cls");
    console::setColor(0xc, 0);
    std::cout <<
        "                                                              __,,,,_\n"
        "                                               _ __..-;''`--/'/ /.',-`-.\n"
        "                                           (`/' ` |  \\ \\ \\\\ / / / / .-'/`,_\n"
        "                                          /'`\\ \\   |  \\ | \\| // // / -.,/_,'-,\n"
        "                                         /<7' ;  \\ \\  | ; ||/ /| | \\/    |`-/,/-.,_,/')\n"
        "                                        /  _.-, `,-\\,__|  _-| / \\ \\/|_/  |    '-/.;.\\'\n"
        "                                        `-`  f/ ;      / __/ \\__ `/ |__/ |\n"
        "                                             `-'      |  -| =|\\_  \\  |-' |\n"
        "                                                   __/   /_..-' `  ),'  //\n"
        "                                               fL ((__.-'((___..-'' \\__.'" << std::endl << std::endl;
    console::setColor(15, 0);
    std::cout << "(Art source : http://www.ascii-art.de/ascii/t/tiger.txt )" << std::endl;
    std::cout << "\nThis program is made with little research and no prior knowledge on this subject. Please manage your expectations." << std::endl; 
    std::cout << "And please don't try to break something as it WILL break." << std::endl;
    std::cout << "\nSource code available at https://github.com/leoTigers/TP2-Universal-Experiment-GA-solver" << std::endl;
    std::cout << "Feel free to experiment and maybe improve this program / fix the mistakes I made :D" << std::endl;
    
    if (argc < 2) {
        std::cout << "\nThis program can run with parameters, to learn more about symbols and parameters, run with -h" << std::endl;
        wait_on_enter();
    }

    // Simulation parameters
    Parameters params{ 0 };
    params = parse_parameters(argc, argv);

    Grid* population = new Grid[params.population_size];

    Grid fittest{};
    
    // Best solution found yet (score 526)
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
    repr(bs);

    int *scores = new int[params.population_size];
    int *scores_indices = new int[params.population_size];

    // CUDA device initialization
    Grid* d_population;
    Parameters* d_params;
    int* d_scores;
    int* d_scores_indices;
    curandState *c_state;
    Grid* d_fittest;

    cudaMalloc((void**)&d_population, params.population_size * sizeof(Grid));
    cudaMalloc((void**)&d_params, sizeof(Parameters));
    cudaMalloc((void**)&d_scores, sizeof(int)* params.population_size);
    cudaMalloc((void**)&d_scores_indices, sizeof(int)* params.population_size);
    cudaMalloc((void**)&c_state, sizeof(curandState)* (params.population_size/THREADS + 1));
    cudaMalloc((void**)&d_fittest, sizeof(Grid));

    time_t start = time(NULL);
    std::cout << "Seed:" << start << std::endl;
    srand(start);

    //initial state
    create_population(params.population_size, population);
    //copy_tab(population[0], bs);
    //population[0].changed = true;

    cudaMemcpy(d_population, population, sizeof(Grid)* params.population_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, &params, sizeof(Parameters), cudaMemcpyHostToDevice);
    setup(c_state, &params);

    time_t t = time(NULL);
    time_t dt;

    for (int iteration = 0; iteration < params.max_iterations; ++iteration) {
        // evalute the population
        cuda_run(params, d_params, population, d_population,
            scores, d_scores, scores_indices, d_scores_indices,
            &fittest, d_fittest, c_state);


        dt = time(NULL);
        if (dt - t >= 2) {
            status(params.population_size, population, iteration, params.max_iterations, fittest, scores, start, t);
            t = dt;
           
        }
    }

    std::cout << "Best fitness : " << -fittest.score << std::endl;
    repr(fittest);

    std::cout << "Seed:" << start << std::endl;
    wait_on_enter();

    return 0;
}
