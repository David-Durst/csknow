//
// Created by durst on 3/31/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_AGGRESSION_DISTRIBUTION_H
#define CSKNOW_INFERENCE_LATENT_AGGRESSION_DISTRIBUTION_H

#include "queries/inference_moments/inference_latent_aggression.h"

namespace csknow::inference_latent_aggression {
    class InferenceLatentAggressionDistributionResult : public QueryResult {
    public:
        const PlayerAtTick & playerAtTick;
        QueryPlayerAtTick & queryPlayerAtTick;
        const InferenceLatentAggressionResult & inferenceLatentAggressionResult;

        explicit InferenceLatentAggressionDistributionResult(const PlayerAtTick & playerAtTick,
                                                             QueryPlayerAtTick & queryPlayerAtTick,
                                                             const InferenceLatentAggressionResult & inferenceLatentAggressionResult) :
                                                             playerAtTick(playerAtTick),
                                                             queryPlayerAtTick(queryPlayerAtTick),
                                                             inferenceLatentAggressionResult(inferenceLatentAggressionResult) {
            variableLength = false;
            startTickColumn = 0;
            ticksPerEvent = 1;
            perTickPlayerLabels = true;
        };

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
            return queryPlayerAtTick.filterByForeignKey(otherTableIndex);
        }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << playerAtTick.tickId[index] << "," << playerAtTick.playerId[index] << "=";

            s << std::fixed << std::setprecision(2);
            bool first = true;
            for (float prob : inferenceLatentAggressionResult.playerAggressionProb[index]) {
                if (!first) {
                    s << " ";
                }
                s << prob;
                first = false;
            }
            s << std::endl;
        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"tick id"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"aggression prob"};
        }
    };
}

#endif //CSKNOW_INFERENCE_LATENT_AGGRESSION_DISTRIBUTION_H
