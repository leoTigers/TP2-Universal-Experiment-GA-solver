#pragma once

#define DIM 7

/* Structures */
enum Object_type {
    ARROW = 0,
    ORB = 1,
    REFLECT = 2
};

enum Arrow_type {
    ONE_USE = 0,
    THREE_USES = 1,
    FIVE_USES = 2,
    INFINITE_USES = 3,
    ROTATING = 4
};

enum Orb_type {
    NORMAL = 0,
    REFRESH = 1
};

typedef struct {
    char type; // 0 Arrow, 1 Orb, 2 Reflect
    char o_type; // Object type : Arrows 0:1T, 1:3T, 2:5T, 3:inf, 4:rot ;; orb: 0:normal, 1:refresh
    char dir; // INITIAL 0 if not type==0, [0;7[ else, starting top-left and going clockwise
    char c_dir; // NOW
    char current_uses;
} Case;

typedef struct {
    Case cases[DIM][DIM];
    char limits[4]; // limits for 5t, inf, rot arrows and refresh
    int changed;
    int score;
    int min_lifetime;
} Grid;

typedef struct {
    int population_size;
    int max_iterations;
    int min_mutations;
    int max_mutations;
    int min_lifetime;
    float retain_rate;
    float mutation_rate;
    float kill_rate;
} Parameters;