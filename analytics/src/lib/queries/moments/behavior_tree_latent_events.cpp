//
// Created by durst on 2/21/23.
//

#include "queries/moments/behavior_tree_latent_events.h"

namespace csknow::behavior_tree_latent_events {
    void BehaviorTreeLatentEvents::runQuery(const string & navPath, VisPoints visPoints,
                                            const MapMeshResult & mapMeshResult, const ReachableResult & reachability,
                                            const DistanceToPlacesResult & distanceToPlaces,
                                            const nearest_nav_cell::NearestNavCell & nearestNavCell,
                                            const Players & players, const Rounds & rounds, const Ticks & ticks,
                                            const PlayerAtTick & playerAtTick,
                                            const WeaponFire & weaponFire, const Hurt & hurt) {

        int numThreads = omp_get_max_threads();
        vector<vector<int64_t>> tmpRoundIds(numThreads);
        vector<vector<int64_t>> tmpRoundStarts(numThreads);
        vector<vector<int64_t>> tmpRoundSizes(numThreads);
        vector<vector<int64_t>> tmpStartTickId(numThreads);
        vector<vector<int64_t>> tmpEndTickId(numThreads);
        vector<vector<int64_t>> tmpLength(numThreads);
        vector<vector<int64_t>> tmpPlayerId(numThreads);
        vector<vector<LatentEventType>> tmpEventType(numThreads);
        TreeThinker defaultThinker{INVALID_ID, AggressiveType::Push};

//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()));
            Blackboard blackboard(navPath, visPoints, mapMeshResult, reachability, distanceToPlaces);
            GlobalQueryNode globalQueryNode(blackboard);
            PlayerQueryNode playerQueryNode(blackboard);

            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                blackboard.streamingManager.update(players, ticks, weaponFire, hurt, playerAtTick, tickIndex,
                                                   nearestNavCell, visPoints);
                globalQueryNode.exec(blackboard.streamingManager.db.batchData.fromNewest(), defaultThinker);

                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                    patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    TreeThinker defaultThinker{playerAtTick.playerId[patIndex], AggressiveType::Push};
                    playerQueryNode.exec(blackboard.streamingManager.db.batchData.fromNewest(), defaultThinker);
                }
            }
        }

    }

}
