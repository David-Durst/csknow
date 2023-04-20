//
// Created by durst on 4/17/23.
//

#include "queries/moments/latent_extractors/engagement_ticks_until_hurt_kill.h"

namespace csknow::latent_engagement {
    void EngagementTicksUntilHurtKill::runQuery(const Rounds &rounds, const Ticks &ticks,
                                                const PlayerAtTick &playerAtTick,
                                                const Hurt &hurt, const Kills &kills,
                                                const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates &behaviorTreeLatentStates) {
        nextHurtId.resize(playerAtTick.size, INVALID_ID);
        nextHurtTickId.resize(playerAtTick.size, INVALID_ID);
        nextKillId.resize(playerAtTick.size, INVALID_ID);
        nextKillTickId.resize(playerAtTick.size, INVALID_ID);
        inEngagement.resize(playerAtTick.size, false);

//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            // forward pass: compute current events
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                 tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                map<int64_t, int64_t> playersHurtingThisTickToEventIndex;
                for (const auto &[_0, _1, hurtIndex]:
                    ticks.hurtPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    if (!isDemoEquipmentAGun(hurt.weapon[hurtIndex])) {
                        continue;
                    }
                    playersHurtingThisTickToEventIndex[hurt.attacker[hurtIndex]] = hurtIndex;
                }
                map<int64_t, int64_t> playersKillingThisTickToEventIndex;
                for (const auto &[_0, _1, killIndex]:
                    ticks.killsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    if (!isDemoEquipmentAGun(static_cast<DemoEquipmentType>(kills.weapon[killIndex]))) {
                        continue;
                    }
                    playersKillingThisTickToEventIndex[kills.killer[killIndex]] = killIndex;
                }
                set<int64_t> playersInEngagementsThisTick;
                for (const auto & [_0, _1, latentEventIndex] :
                    behaviorTreeLatentStates.eventsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                    csknow::behavior_tree_latent_states::EngagementStatePayload engagementStatePayload =
                        std::get<csknow::behavior_tree_latent_states::EngagementStatePayload>(
                            behaviorTreeLatentStates.statePayload[latentEventIndex]);
                    if (behaviorTreeLatentStates.latentStateType[latentEventIndex] ==
                        csknow::behavior_tree_latent_states::LatentStateType::Engagement) {
                        playersInEngagementsThisTick.insert(engagementStatePayload.sourceId);
                    }
                }

                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    int64_t playerId = playerAtTick.playerId[patIndex];

                    if (playersHurtingThisTickToEventIndex.find(playerId) != playersHurtingThisTickToEventIndex.end()) {
                        nextHurtId[patIndex] = playersHurtingThisTickToEventIndex[playerId];
                        nextHurtTickId[patIndex] = tickIndex;
                    }
                    if (playersKillingThisTickToEventIndex.find(playerId) != playersKillingThisTickToEventIndex.end()) {
                        nextKillId[patIndex] = playersKillingThisTickToEventIndex[playerId];
                        nextKillTickId[patIndex] = tickIndex;
                    }
                    inEngagement[patIndex] =
                        playersInEngagementsThisTick.find(playerId) != playersInEngagementsThisTick.end();
                }
            }

            map<int64_t, int64_t> playerToNextHurtId, playerToNextHurtTickId, playerToNextKillId, playerToNextKillTickId;
            // backward pass: compute future
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].maxId;
                 tickIndex >= rounds.ticksPerRound[roundIndex].minId; tickIndex--) {
                for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                     patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                    int64_t playerId = playerAtTick.playerId[patIndex];

                    if (nextHurtId[patIndex] != INVALID_ID) {
                        playerToNextHurtId[playerId] = nextHurtId[patIndex];
                        playerToNextHurtTickId[playerId] = nextHurtTickId[patIndex];
                    }
                    else if (playerToNextHurtId.find(playerId) != playerToNextHurtId.end()) {
                        nextHurtId[patIndex] = playerToNextHurtId[playerId];
                        nextHurtTickId[patIndex] = playerToNextHurtTickId[playerId];
                    }

                    if (nextKillId[patIndex] != INVALID_ID) {
                        playerToNextKillId[playerId] = nextKillId[patIndex];
                        playerToNextKillTickId[playerId] = nextKillTickId[patIndex];
                    }
                    else if (playerToNextKillId.find(playerId) != playerToNextKillId.end()) {
                        nextKillId[patIndex] = playerToNextKillId[playerId];
                        nextKillTickId[patIndex] = playerToNextKillTickId[playerId];
                    }
                }
            }
        }

        size = playerAtTick.size;
    }
}