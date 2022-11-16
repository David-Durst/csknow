//
// Created by durst on 11/15/22.
//
#include "queries/moments/fire_history.h"
#include <omp.h>

namespace csknow::fire_history {
    void FireHistoryResult::runQuery(const WeaponFire &weaponFire, const PlayerAtTick &playerAtTick) {
        int numThreads = omp_get_max_threads();
        vector<vector<int64_t>> tmpTickId(numThreads);
        vector<vector<int64_t>> tmpPlayerId(numThreads);
        vector<vector<int16_t>> tmpTicksSinceLastFire(numThreads);
        vector<vector<int64_t>> tmpLastShotFiredTickId(numThreads);
        vector<vector<int16_t>> tmpTicksUntilNextFire(numThreads);
        vector<vector<int64_t>> tmpNextShotFiredTickId(numThreads);
        vector<vector<int64_t>> tmpHoldingAttackButton(numThreads);
    }
}
