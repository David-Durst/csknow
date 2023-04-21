//
// Created by durst on 3/31/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_ENGAGEMENT_DISTRIBUTION_H
#define CSKNOW_INFERENCE_LATENT_ENGAGEMENT_DISTRIBUTION_H

#include "queries/inference_moments/inference_latent_engagement.h"

namespace csknow::inference_latent_engagement {
    class InferenceLatentEngagementDistributionResult : public QueryResult {
    public:
        const PlayerAtTick & playerAtTick;
        QueryPlayerAtTick & queryPlayerAtTick;
        const InferenceLatentEngagementResult & inferenceLatentEngagementResult;

        explicit InferenceLatentEngagementDistributionResult(const PlayerAtTick & playerAtTick,
                                                             QueryPlayerAtTick & queryPlayerAtTick,
                                                             const InferenceLatentEngagementResult & inferenceLatentEngagementResult) :
            playerAtTick(playerAtTick),
            queryPlayerAtTick(queryPlayerAtTick),
            inferenceLatentEngagementResult(inferenceLatentEngagementResult) {
            variableLength = false;
            startTickColumn = 0;
            ticksPerEvent = 1;
            perTickPlayerLabels = true;
        };

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
            return queryPlayerAtTick.filterByForeignKey(otherTableIndex);
        }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << playerAtTick.tickId[index] << ",";

            s << std::fixed << std::setprecision(2);
            bool first = true;
            for (const auto prob : inferenceLatentEngagementResult.playerEngageProbs[index]) {
                if (!first) {
                    s << ";";
                }
                s << prob.playerId << "=" << prob.prob;
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
            return {"engage prob"};
        }
    };
}

#endif //CSKNOW_INFERENCE_LATENT_ENGAGEMENT_DISTRIBUTION_H
