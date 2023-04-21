//
// Created by durst on 4/20/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_AREA_DISTRIBUTION_H
#define CSKNOW_INFERENCE_LATENT_AREA_DISTRIBUTION_H

#include "queries/inference_moments/inference_latent_area.h"
#include "queries/base_tables.h"

namespace csknow::inference_latent_area {
    class InferenceLatentAreaDistributionResult : public QueryResult {
    public:
        const PlayerAtTick & playerAtTick;
        QueryPlayerAtTick & queryPlayerAtTick;
        const DistanceToPlacesResult & distanceToPlacesResult;
        const InferenceLatentAreaResult & inferenceLatentAreaResult;
        array<Vec3, csknow::feature_store::area_grid_size> areaLabelPositions;

        void setLabelPositions();

        explicit InferenceLatentAreaDistributionResult(const PlayerAtTick & playerAtTick,
                                                        QueryPlayerAtTick & queryPlayerAtTick,
                                                        const DistanceToPlacesResult & distanceToPlacesResult,
                                                        const InferenceLatentAreaResult & inferenceLatentAreaResult) :
                playerAtTick(playerAtTick),
                queryPlayerAtTick(queryPlayerAtTick),
                distanceToPlacesResult(distanceToPlacesResult),
                inferenceLatentAreaResult(inferenceLatentAreaResult) {
            setLabelPositions();
            variableLength = false;
            startTickColumn = 0;
            ticksPerEvent = 1;
            perTickPosLabels = true;
            for (size_t i = 0; i < areaLabelPositions.size(); i++) {
                posLabelsPositions.push_back(areaLabelPositions[i].toCSV("_"));
            }
            havePerTickPosOffsets = true;
            perTickPosOffsetsColumn = 1
        };

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
            return queryPlayerAtTick.filterByForeignKey(otherTableIndex);
        }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << playerAtTick.tickId[index] << ",";

            s << std::setprecision(2);
            bool first = true;
            for (const auto prob : inferenceLatentAreaResult.playerAreaProb[index]) {
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
            return {"area prob", "place offset"};
        }
    };
}

#endif //CSKNOW_INFERENCE_LATENT_AREA_DISTRIBUTION_H
