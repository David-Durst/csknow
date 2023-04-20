//
// Created by durst on 4/9/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_PLACE_DISTRIBUTION_H
#define CSKNOW_INFERENCE_LATENT_PLACE_DISTRIBUTION_H

#include "queries/inference_moments/inference_latent_place.h"
#include "queries/base_tables.h"

namespace csknow::inference_latent_place {
    class InferenceLatentPlaceDistributionResult : public QueryResult {
    public:
        const PlayerAtTick & playerAtTick;
        QueryPlayerAtTick & queryPlayerAtTick;
        const DistanceToPlacesResult & distanceToPlacesResult;
        const InferenceLatentPlaceResult & inferenceLatentPlaceResult;
        array<Vec3, csknow::feature_store::num_places> placeLabelPositions;

        void setLabelPositions();

        explicit InferenceLatentPlaceDistributionResult(const PlayerAtTick & playerAtTick,
                                                        QueryPlayerAtTick & queryPlayerAtTick,
                                                        const DistanceToPlacesResult & distanceToPlacesResult,
                                                        const InferenceLatentPlaceResult & inferenceLatentPlaceResult) :
            playerAtTick(playerAtTick),
            queryPlayerAtTick(queryPlayerAtTick),
            distanceToPlacesResult(distanceToPlacesResult),
            inferenceLatentPlaceResult(inferenceLatentPlaceResult) {
            setLabelPositions();
            variableLength = false;
            startTickColumn = 0;
            ticksPerEvent = 1;
            perTickPosLabels = true;
            for (size_t i = 0; i < placeLabelPositions.size(); i++) {
                posLabelsPositions.push_back(placeLabelPositions[i].toCSV("_"));
            }
        };

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
            return queryPlayerAtTick.filterByForeignKey(otherTableIndex);
        }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << playerAtTick.tickId[index] << ",";

            s << std::setprecision(2);
            bool first = true;
            for (const auto prob : inferenceLatentPlaceResult.playerPlaceProb[index]) {
                if (!first) {
                    s << ";";
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
            return {"place prob"};
        }
    };
}

#endif //CSKNOW_INFERENCE_LATENT_PLACE_DISTRIBUTION_H
