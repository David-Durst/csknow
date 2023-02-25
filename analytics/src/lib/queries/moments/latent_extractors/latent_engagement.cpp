//
// Created by durst on 2/24/23.
//

#include "queries/moments/latent_extractors/latent_engagement.h"
#include "indices/build_indexes.h"


namespace csknow::latent_engagement {
    void LatentEngagementResult::runQuery(const Rounds &rounds, const Ticks &ticks, const Hurt &hurt,
                                          const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates &behaviorTreeLatentStates) {
        int numThreads = omp_get_max_threads();
        vector<vector<int64_t>> tmpRoundIds(numThreads);
        vector<vector<int64_t>> tmpRoundStarts(numThreads);
        vector<vector<int64_t>> tmpRoundSizes(numThreads);
        vector<vector<int64_t>> tmpStartTickId(numThreads);
        vector<vector<int64_t>> tmpEndTickId(numThreads);
        vector<vector<int64_t>> tmpLength(numThreads);
        vector<vector<vector<int64_t>>> tmpPlayerId(numThreads);
        vector<vector<vector<EngagementRole>>> tmpRole(numThreads);
        vector<vector<vector<int64_t>>> tmpHurtTickIds(numThreads);
        vector<vector<vector<int64_t>>> tmpHurtIds(numThreads);

        // for each round
        // track events for each pairs of player.
        // start a new event for a pair when hurt event with no prior one or far away prior one
        // clear out all hurt events on end of round
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int threadNum = omp_get_thread_num();
            tmpRoundIds[threadNum].push_back(roundIndex);
            tmpRoundStarts[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()));

            map<int64_t, int64_t> latentEventIndexToTmpEngagementIndex;

            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                map<pair<int64_t, int64_t>, int64_t> hurtAttackerAndVictimToHurtId;
                for (const auto & [_0, _1, hurtIndex] :
                    ticks.hurtPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    if (!isDemoEquipmentAGun(hurt.weapon[hurtIndex])) {
                        continue;
                    }
                    hurtAttackerAndVictimToHurtId[{hurt.attacker[hurtIndex], hurt.victim[hurtIndex]}] = hurtIndex;
                }

                for (const auto & [_0, _1, latentEventIndex] :
                    behaviorTreeLatentStates.eventsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    //std::cout << "tick index " << tickIndex << " in latent event " << latentEventIndex <<
                    //    " start index" << behaviorTreeLatentStates.startTickId[latentEventIndex] << std::endl;

                    if (behaviorTreeLatentStates.latentStateType[latentEventIndex] ==
                        csknow::behavior_tree_latent_states::LatentStateType::Engagement) {

                        csknow::behavior_tree_latent_states::EngagementStatePayload engagementStatePayload =
                            std::get<csknow::behavior_tree_latent_states::EngagementStatePayload>(
                                behaviorTreeLatentStates.statePayload[latentEventIndex]);

                        if (behaviorTreeLatentStates.startTickId[latentEventIndex] == tickIndex &&
                            behaviorTreeLatentStates.latentStateType[latentEventIndex] ==
                            csknow::behavior_tree_latent_states::LatentStateType::Engagement) {
                            //std::cout << "adding latent event" << std::endl;
                            latentEventIndexToTmpEngagementIndex[latentEventIndex] = tmpStartTickId.size();
                            tmpStartTickId[threadNum].push_back(behaviorTreeLatentStates.startTickId[latentEventIndex]);
                            tmpEndTickId[threadNum].push_back(behaviorTreeLatentStates.endTickId[latentEventIndex]);
                            tmpLength[threadNum].push_back(behaviorTreeLatentStates.tickLength[latentEventIndex]);
                            tmpPlayerId[threadNum].push_back({engagementStatePayload.sourceId, engagementStatePayload.targetId});
                            tmpRole[threadNum].push_back({EngagementRole::Attacker, EngagementRole::Victim});
                            tmpHurtTickIds[threadNum].push_back({});
                            tmpHurtIds[threadNum].push_back({});
                        }

                        if (hurtAttackerAndVictimToHurtId.find({engagementStatePayload.sourceId, engagementStatePayload.targetId}) !=
                            hurtAttackerAndVictimToHurtId.end()) {
                            int64_t tmpIndex = latentEventIndexToTmpEngagementIndex[latentEventIndex];
                            tmpHurtTickIds[threadNum][tmpIndex].push_back(tickIndex);
                            tmpHurtIds[threadNum][tmpIndex].push_back(
                                hurtAttackerAndVictimToHurtId[{engagementStatePayload.sourceId, engagementStatePayload.targetId}]);
                        }
                    }

                }
            }

            tmpRoundSizes[threadNum].push_back(static_cast<int64_t>(tmpStartTickId[threadNum].size()) - tmpRoundStarts[threadNum].back());
        }

        mergeThreadResults(numThreads, rowIndicesPerRound, tmpRoundIds, tmpRoundStarts, tmpRoundSizes,
                           startTickId, size,
                           [&](int64_t minThreadId, int64_t tmpRowId) {
                               startTickId.push_back(tmpStartTickId[minThreadId][tmpRowId]);
                               endTickId.push_back(tmpEndTickId[minThreadId][tmpRowId]);
                               tickLength.push_back(tmpLength[minThreadId][tmpRowId]);
                               playerId.push_back(tmpPlayerId[minThreadId][tmpRowId]);
                               role.push_back(tmpRole[minThreadId][tmpRowId]);
                               hurtTickIds.push_back(tmpHurtTickIds[minThreadId][tmpRowId]);
                               hurtIds.push_back(tmpHurtIds[minThreadId][tmpRowId]);
                           });
        vector<const int64_t *> foreignKeyCols{startTickId.data(), endTickId.data()};
        engagementsPerTick = buildIntervalIndex(foreignKeyCols, size);
    }
}
