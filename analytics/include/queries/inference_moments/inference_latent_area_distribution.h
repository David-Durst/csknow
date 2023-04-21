//
// Created by durst on 4/20/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_AREA_DISTRIBUTION_H
#define CSKNOW_INFERENCE_LATENT_AREA_DISTRIBUTION_H

#include "queries/inference_moments/inference_latent_area.h"
#include "queries/inference_moments/inference_latent_place.h"
#include "queries/base_tables.h"

namespace csknow::inference_latent_area {
    class InferenceLatentAreaDistributionResult : public QueryResult {
    public:
        const PlayerAtTick & playerAtTick;
        QueryPlayerAtTick & queryPlayerAtTick;
        const DistanceToPlacesResult & distanceToPlacesResult;
        const InferenceLatentAreaResult & inferenceLatentAreaResult;
        const csknow::inference_latent_place::InferenceLatentPlaceResult & inferenceLatentPlaceResult;

        explicit InferenceLatentAreaDistributionResult(const PlayerAtTick & playerAtTick,
                                                        QueryPlayerAtTick & queryPlayerAtTick,
                                                        const DistanceToPlacesResult & distanceToPlacesResult,
                                                        const InferenceLatentAreaResult & inferenceLatentAreaResult,
                                                        const csknow::inference_latent_place::InferenceLatentPlaceResult &
                                                            inferenceLatentPlaceResult) :
                playerAtTick(playerAtTick),
                queryPlayerAtTick(queryPlayerAtTick),
                distanceToPlacesResult(distanceToPlacesResult),
                inferenceLatentAreaResult(inferenceLatentAreaResult),
                inferenceLatentPlaceResult(inferenceLatentPlaceResult) {
            variableLength = false;
            startTickColumn = 0;
            ticksPerEvent = 1;
            perTickPosLabels = true;
            havePerTickPos = true;
            perTickPosAABBColumn = 1;
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
            const string & place = distanceToPlacesResult.places[inferenceLatentPlaceResult.placeIndex[index]];
            const AABB & placeAABB = distanceToPlacesResult.placeToAABB.at(place);
            s << "," << placeAABB.min.toCSV("_") << ";" << placeAABB.max.toCSV("_");
            s << std::endl;
        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"tick id"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"area prob", "place aabb"};
        }
    };
}

#endif //CSKNOW_INFERENCE_LATENT_AREA_DISTRIBUTION_H
