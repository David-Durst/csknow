#include "indices/spotted.h"
#include "omp.h"

SpottedIndex::SpottedIndex(const Position & position, const Spotted & spotted) {
    int64_t numGames = spotted.gameStarts.size() - 1;
    int numThreads = omp_get_max_threads();
    vector<bool> tmpVisible[numThreads][NUM_PLAYERS][NUM_PLAYERS];
    vector<int64_t> tmpIndices[numThreads][NUM_PLAYERS][NUM_PLAYERS];
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

        map<string, int> playerNameToIndex = position.getPlayerNameToIndex(gameIndex);
        int64_t spottedIndex = spotted.gameStarts[gameIndex];
        // need position as outer loop since since demoTick is unique only for position, may have multiple spotted
        // for one position tick
        for (int64_t positionIndex = position.gameStarts[gameIndex];
             positionIndex < position.gameStarts[gameIndex + 1];
             positionIndex++) {
            while (spottedIndex < spotted.size &&
                   spotted.demoFile[spottedIndex] == position.demoFile[positionIndex] &&
                   spotted.demoTickNumber[spottedIndex] <= position.demoTickNumber[positionIndex]) {
                // skip invalid rows
                if (spotted.skipRows.find(spottedIndex) != spotted.skipRows.end()) {
                    spottedIndex++;
                    continue;
                }
                int spottedPlayer = playerNameToIndex[spotted.spottedPlayer[spottedIndex]];
                for (int j = 0; j < NUM_PLAYERS; j++) {
                    lastVisible[spottedPlayer][j] = spotted.spotters[j].playerSpotter[spottedIndex];
                }
                spottedIndex++;
            }

            for (int i = 0; i < NUM_PLAYERS; i++) {
                for (int j = 0; j < NUM_PLAYERS; j++) {
                    tmpVisible[threadNum][i][j].push_back(lastVisible[i][j]);
                    tmpIndices[threadNum][i][j].push_back(positionIndex);
                }
            }
        }
    }

    for (int i = 0; i < NUM_PLAYERS; i++) {
        for (int j = 0; j < NUM_PLAYERS; j++) {
            visible[i][j].resize(position.size);
        }
    }

    for (int thread = 0; thread < numThreads; thread++) {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            for (int j = 0; j < NUM_PLAYERS; j++) {
                for (int t = 0; t < tmpVisible[thread][i][j].size(); t++) {
                    visible[i][j][tmpIndices[thread][i][j][t]] = tmpVisible[thread][i][j][t];
                }
            }
        }
    }
}

