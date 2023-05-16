//
// Created by durst on 2/26/23.
//

#include "queries/grenade/smoke_grenade.h"
#include "indices/build_indexes.h"
#include <omp.h>
#include <map>

namespace csknow::smoke_grenade {
    void SmokeGrenadeResult::runQuery(const Rounds & rounds, const Ticks & ticks, const Grenades & grenades,
                                      const GrenadeTrajectories & grenadeTrajectories) {
        int numThreads = omp_get_max_threads();
        vector<vector<int64_t>> tmpRoundIds(numThreads);
        vector<vector<int64_t>> tmpRoundStarts(numThreads);
        vector<vector<int64_t>> tmpRoundSizes(numThreads);
        vector<vector<int64_t>> tmpTickId(numThreads);
        vector<vector<int64_t>> tmpThrowerId(numThreads);
        vector<vector<SmokeGrenadeState>> tmpState(numThreads);
        vector<vector<Vec3>> tmpPos(numThreads);

//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()));

            std::map<int64_t, std::map<int64_t, Vec3>> grenadeToTickToPos;
            std::map<int64_t, Vec3> grenadeToLastPos;

            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {

                for (const auto & [_0, _1, grenadeIndex] :
                    ticks.grenadesPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    if (static_cast<DemoEquipmentType>(grenades.grenadeType[grenadeIndex]) !=
                        DemoEquipmentType::EqSmoke) {
                        continue;
                    }
                    tmpTickId[threadNum].push_back(tickIndex);
                    tmpThrowerId[threadNum].push_back(grenades.thrower[grenadeIndex]);
                    if (tickIndex < grenades.activeTick[grenadeIndex]) {
                        tmpState[threadNum].push_back(SmokeGrenadeState::Thrown);
                    }
                    else {
                        tmpState[threadNum].push_back(SmokeGrenadeState::Active);
                    }

                    // if first tick with grenade, add it to map
                    if (grenadeToTickToPos.find(grenadeIndex) == grenadeToTickToPos.end()) {
                        size_t trajectoryOffset = 0;
                        auto trajectory = grenades.trajectoryPerGrenade.intervalToEvent.findOverlapping(grenadeIndex, grenadeIndex);
                        std::sort(trajectory.begin(), trajectory.end(), [](const Interval<int64_t, int64_t> & a, const Interval<int64_t, int64_t> & b) {
                            return a.value < b.value;
                        });
                        for (const auto & [_0, _1, grenadeTrajectoryIndex] : trajectory) {
                            Vec3 curPos {
                                grenadeTrajectories.posX[grenadeTrajectoryIndex],
                                grenadeTrajectories.posY[grenadeTrajectoryIndex],
                                grenadeTrajectories.posZ[grenadeTrajectoryIndex]
                            };
                            grenadeToTickToPos[grenadeIndex][tickIndex + trajectoryOffset] = curPos;
                            grenadeToLastPos[grenadeIndex] = curPos;
                            trajectoryOffset++;
                        }
                    }

                    // smoke grenade trajectories last less than total throw time (seem to stop when grenade comes to rest)
                    // so repeat pos once past length
                    if (grenadeToTickToPos[grenadeIndex].find(tickIndex) != grenadeToTickToPos[grenadeIndex].end()) {
                        tmpPos[threadNum].push_back(grenadeToTickToPos[grenadeIndex][tickIndex]);
                    }
                    else {
                        tmpPos[threadNum].push_back(grenadeToLastPos[grenadeIndex]);
                    }
                }
            }

            tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
        }
        mergeThreadResults(numThreads, rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                           tickId, size,
                           [&](int64_t minThreadId, int64_t tmpRowId) {
                               tickId.push_back(tmpTickId[minThreadId][tmpRowId]);
                               throwerId.push_back(tmpThrowerId[minThreadId][tmpRowId]);
                               state.push_back(tmpState[minThreadId][tmpRowId]);
                               pos.push_back(tmpPos[minThreadId][tmpRowId]);
                           });
        vector<std::reference_wrapper<const vector<int64_t>>> foreignKeyCols{tickId};
        smokeGrenadesPerTick = buildIntervalIndex(foreignKeyCols, size);
    }
}
