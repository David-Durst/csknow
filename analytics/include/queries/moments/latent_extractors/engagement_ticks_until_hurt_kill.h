//
// Created by durst on 4/17/23.
//

#ifndef CSKNOW_ENGAGEMENT_TICKS_UNTIL_HURT_KILL_H
#define CSKNOW_ENGAGEMENT_TICKS_UNTIL_HURT_KILL_H

#include "queries/moments/latent_extractors/latent_engagement.h"

namespace csknow::latent_engagement {
    class EngagementTicksUntilHurtKill : QueryResult {
    public:
        vector<RangeIndexEntry> rowIndicesPerRound;
        vector<int64_t> nextHurtId;
        vector<int64_t> nextHurtTickId;
        vector<int64_t> nextKillId;
        vector<int64_t> nextKillTickId;
        vector<bool> inEngagement;
        IntervalIndex engagementTicksUntilHurtKillPerTick;

        EngagementTicksUntilHurtKill() {
            variableLength = false;
            startTickColumn = 0;
            ticksPerEvent = 1;
        }

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
            vector<int64_t> result;
            // Normally would segfault, but this query is slow so I don't run for all rounds in debug cases
            if (otherTableIndex >= static_cast<int64_t>(rowIndicesPerRound.size())) {
                return result;
            }
            for (int64_t i = rowIndicesPerRound[otherTableIndex].minId; i <= rowIndicesPerRound[otherTableIndex].maxId; i++) {
                if (i == -1) {
                    continue;
                }
                result.push_back(i);
            }
            return result;
        }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << nextHurtId[index] << "," << nextKillId[index] <<
                "," << inEngagement[index] << std::endl;
        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"next hurt id", "next kill id"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"in engagement"};
        }

        void runQuery(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                      const Hurt & hurt, const Kills & kills,
                      const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates &behaviorTreeLatentStates);
    };
}

#endif //CSKNOW_ENGAGEMENT_TICKS_UNTIL_HURT_KILL_H
