#ifndef CSKNOW_INDICES_H
#define CSKNOW_INDICES_H
#include "load_data.h"
#include "omp.h"

class SpottedIndex {
public:
    // visible[i][j] - is i visible to j
    bool * visible[NUM_PLAYERS][NUM_PLAYERS];

    SpottedIndex(int64_t rows, const Position & position, const Spotted & spotted) {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            for (int j = 0; j < NUM_PLAYERS; j++) {
                free(visible[i][j]);
                visible[i][j] = (bool *) malloc(rows * sizeof(bool));
            }
        }

        int64_t numGames = spotted.gameStarts.size() - 1;
        int numThreads = omp_get_max_threads();
        vector<bool> tmpVisible[numThreads][NUM_PLAYERS][NUM_PLAYERS];
#pragma omp parallel for
        for (int64_t gameIndex = 0; gameIndex < numGames; gameIndex++) {
            int threadNum = omp_get_thread_num();
            // initialize as not seeing anyone
            bool lastVisible[NUM_PLAYERS][NUM_PLAYERS];
            for (int i = 0; i < NUM_PLAYERS; i++) {
                for (int j = 0; j < NUM_PLAYERS; j++) {
                    lastVisible[i][j] = false;
                }
            }

            int64_t spottedIndex = spotted.gameStarts[gameIndex];
            // need position as outer loop since since demoTick is unique only for position, may have multiple spotted
            // for one position tick
            for (int64_t positionIndex = position.gameStarts[gameIndex]; positionIndex < position.gameStarts[gameIndex + 1];
                    positionIndex++) {
                while (spottedIndex < spotted.size && spotted.demoFile[spottedIndex] == position.demoFile[positionIndex] &&
                       spotted.demoTickNumber[spottedIndex] <= position.demoTickNumber[positionIndex]) {
                    // skip invalid rows
                    if (spotted.skipRows.find(spottedIndex) != spotted.skipRows.end()) {
                        spottedIndex++;
                        continue;
                    }
                    int spottedPlayer = playerNameToIndex[spotted.spottedPlayer[spottedIndex]];
                    for (int i = 0; i < NUM_PLAYERS; i++) {
                        spottedPerWindow[spottedPlayer][i] = spotted.spotters[i].playerSpotter[spottedIndex];
                    }
                    spottedIndex++;
                }

                for (int i = 0; i < NUM_PLAYERS; i++) {
                    for (int j = 0; j < NUM_PLAYERS; j++) {
                        tmpVisible[threadNum][i][j].push_back(lastVisible[i][j]);
                    }
                }
            }

            int64_t positionIndex = 0;
            for (int64_t spottedIndex = spotted.gameStarts[gameIndex];
                    spottedIndex < spotted.size && spottedIndex < spotted.gameStarts[gameIndex + 1];
                    spottedIndex++) {
                while (position.demoTickNumber[positionIndex] <= spotted.demoTickNumber[spottedIndex] &&
                        position.demoFile[positionIndex] == spotted.demoFile[spottedIndex]) {
                    if (position.demoTickNumber[positionIndex] == spotted.demoTickNumber[spottedIndex]) {

                    }
                    for (int i = 0; i < NUM_PLAYERS; i++) {
                        for (int j = 0; j < NUM_PLAYERS; j++) {
                            tmpVisible[threadNum][i][j].push_back(lastVisible[i][j]);
                        }
                    }
                }
                // skip invalid rows
                if (spotted.skipRows.find(spottedIndex) != spotted.skipRows.end()) {
                    spottedIndex++;
                    continue;
                }
                int spottedPlayer = playerNameToIndex[spotted.spottedPlayer[spottedIndex]];
                for (int i = 0; i < NUM_PLAYERS; i++) {
                    spottedPerWindow[spottedPlayer][i] = spotted.spotters[i].playerSpotter[spottedIndex];
                }
                spottedIndex++;
            }
        }
    }

    ~SpottedIndex() {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            for (int j = 0; j < NUM_PLAYERS; j++) {
                free(visible[i][j]);
            }
        }
    }

    SpottedIndex(const SpottedIndex& other) = delete;
    SpottedIndex& operator=(const SpottedIndex& other) = delete;
};

#endif //CSKNOW_INDICES_H
