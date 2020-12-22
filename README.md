This program is made with little research and no prior knowledge on this subject. Please manage your expectations.
And please don't try to break something as it WILL break.

Source code available at https://github.com/leoTigers/TP2-Universal-Experiment-GA-solver
Feel free to experiment and maybe improve this program / fix the mistakes I made :D

This program can run with parameters, to learn more about symbols and parameters, run with -h

Usage: Universal_experiment_genetic_alg <option(s)>
Options:
        -h,--help               Show this help message
        -p,--population POPULATION_SIZE Specify the size of the population
        -i,--iterations ITERATION_COUNT Specify the number of iterations
        -m,--min-mutation MIN_MUTATIONS Specify the minimum of mutations that WILL occur
        -M,--max-mutation MAX_MUTATIONS Specify the maxiimum of mutations that CAN occur
        -rr,--retain RETAIN_RATE        Specify the retain rate (0.2 means top 20% will survive at the end of a round)
        -mr,--mutation MUTATION_RATE    Specify the mutation rate (between 0 and 1)


Symbols: (to see colors, start program with option -h)
        ┌ ^ ┐ > ┘ v └ <  : one use arrows
        ┌ ^ ┐ > ┘ v └ <  : three uses arrows
        ┌ ^ ┐ > ┘ v └ <  : five uses arrows
        ┌ ^ ┐ > ┘ v └ <  : infinite uses arrows
        ┌ ^ ┐ > ┘ v └ <  : rotating arrows
        O : normal orb
        O : refresh orb
        R : reflect