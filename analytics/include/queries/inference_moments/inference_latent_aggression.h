//
// Created by durst on 3/5/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_AGGRESSION_H
#define CSKNOW_INFERENCE_LATENT_AGGRESSION_H
#include "queries/moments/behavior_tree_latent_states.h"

namespace csknow::inference_latent_aggression {
    class InferenceLatentAggressionResult : public QueryResult {
    public:
        const PlayerAtTick & playerAtTick;
        vector<RangeIndexEntry> rowIndicesPerRound;
        vector<int64_t> startTickId;
        vector<int64_t> endTickId;
        vector<int64_t> tickLength;
        vector<vector<int64_t>> playerId;
        vector<vector<feature_store::NearestEnemyState>> role;
        IntervalIndex aggressionPerTick;
        vector<array<float, feature_store::numNearestEnemyState>> playerAggressionProb;

        explicit InferenceLatentAggressionResult(const PlayerAtTick & playerAtTick) : playerAtTick(playerAtTick) {
            variableLength = true;
            startTickColumn = 0;
            perEventLengthColumn = 2;
            havePlayerLabels = true;
            playerLabels = {"F", "C", "B"};
            playersToLabelColumn = 0;
            playerLabelIndicesColumn = 1;
        };

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
            vector<int64_t> result;
            for (int64_t i = rowIndicesPerRound[otherTableIndex].minId; i <= rowIndicesPerRound[otherTableIndex].maxId; i++) {
                if (i == -1) {
                    continue;
                }
                result.push_back(i);
            }
            return result;
        }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << startTickId[index] << "," << endTickId[index] << "," << tickLength[index] << ",";

            vector<string> tmp;
            for (int64_t pId : playerId[index]) {
                tmp.push_back(std::to_string(pId));
            }
            commaSeparateList(s, tmp, ";");
            s << ",";

            tmp.clear();
            for (feature_store::NearestEnemyState r : role[index]) {
                tmp.push_back(std::to_string(enumAsInt(r)));
            }
            commaSeparateList(s, tmp, ";");

            s << std::endl;
        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"start tick id", "end tick id", "length"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"player ids", "roles"};
        }

        void runQuery(const string & modelsDir, const Rounds & rounds, const Ticks & ticks,
                      const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates & behaviorTreeLatentStates);

    };
}

#endif //CSKNOW_INFERENCE_LATENT_AGGRESSION_H
