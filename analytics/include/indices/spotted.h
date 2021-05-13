#ifndef CSKNOW_SPOTTED_H
#define CSKNOW_SPOTTED_H
#include "load_data.h"
#include "omp.h"

class SpottedIndex {
public:
    // visible[i][j] - is i visible to j
    vector<bool> visible[NUM_PLAYERS][NUM_PLAYERS];
    SpottedIndex(const Ticks & ticks, const PlayerAtTick & pat, const Spotted & spotted);
};

#endif //CSKNOW_SPOTTED_H
