//
// Created by durst on 8/3/23.
//

#include "bots/analysis/humanness_metrics.h"

namespace csknow::humanness_metrics {
    HumannessMetrics::HumannessMetrics(const csknow::feature_store::TeamFeatureStoreResult &teamFeatureStoreResult,
                                       const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                       const Hurt & hurt, const WeaponFire & weaponFire) {
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                if (teamFeatureStoreResult.nonDecimatedValidRetakeTicks[tickIndex]) {
                    vector<int64_t> attackersThisTick;
                    for (const auto & [_0, _1, weaponFireIndex] :
                        ticks.weaponFirePerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                        attackersThisTick.push_back(weaponFire.shooter[weaponFireIndex]);
                    }

                    vector<int64_t> victimsThisTick;
                    for (const auto & [_0, _1, hurtIndex] :
                            ticks.hurtPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                        victimsThisTick.push_back(hurt.victim[hurtIndex]);
                    }
                }
            }
        }
    }
}