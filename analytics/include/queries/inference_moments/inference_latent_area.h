//
// Created by durst on 4/20/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_AREA_H
#define CSKNOW_INFERENCE_LATENT_AREA_H

#include "load_data.h"
#include "queries/query.h"
#include "geometry.h"
#include "enum_helpers.h"
#include "bots/analysis/feature_store.h"
#include "queries/moments/behavior_tree_latent_states.h"
#include "queries/inference_moments/inference_latent_area_helpers.h"

namespace csknow::inference_latent_area {

    class InferenceLatentAreaResult : public QueryResult {
    public:
        vector<RangeIndexEntry> rowIndicesPerRound;
        vector<int64_t> startTickId;
        vector<int64_t> endTickId;
        vector<int64_t> tickLength;
        vector<int64_t> playerId;
        vector<size_t> areaGridIndex;
        vector<array<float, csknow::feature_store::num_places>> playerAreaProb;
        IntervalIndex areasPerTick;

        InferenceLatentAreaResult() {
            variableLength = true;
            startTickColumn = 0;
            perEventLengthColumn = 2;
            havePlayerLabels = true;
            playerLabels = {};
            for (size_t areaIndex = 0; areaIndex < csknow::feature_store::area_grid_size; areaIndex++) {
                playerLabels.push_back(
                        std::to_string(areaIndex % csknow::feature_store::area_grid_dim) + "," +
                        std::to_string(areaIndex / csknow::feature_store::area_grid_dim)
                );
            }
            playersToLabelColumn = 0;
            playerLabelIndicesColumn = 1;
        }

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
            vector<int64_t> result;
            // Normally would segfault, but this query is slow so I don't run for all rounds in debug cases
            if (otherTableIndex >= static_cast<int64_t>(rowIndicesPerRound.size())) {
                return result;
            }
            for (int64_t i = rowIndicesPerRound[otherTableIndex].minId;
                 i <= rowIndicesPerRound[otherTableIndex].maxId; i++) {
                if (i == -1) {
                    continue;
                }
                result.push_back(i);
            }
            return result;
        }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << startTickId[index] << "," << endTickId[index] << "," << tickLength[index]
              << "," << playerId[index] << "," << areaGridIndex[index] << std::endl;
        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"start tick id", "end tick id", "length"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"player id", "area grid index"};
        }

        void toHDF5Inner(HighFive::File &file) override {

            HighFive::DataSetCreateProps hdf5FlatCreateProps;
            hdf5FlatCreateProps.add(HighFive::Deflate(6));
            hdf5FlatCreateProps.add(HighFive::Chunking(startTickId.size()));

            file.createDataSet("/data/start tick id", startTickId, hdf5FlatCreateProps);
            file.createDataSet("/data/end tick id", endTickId, hdf5FlatCreateProps);
            file.createDataSet("/data/tick length", tickLength, hdf5FlatCreateProps);
            file.createDataSet("/data/player id", playerId, hdf5FlatCreateProps);
            file.createDataSet("/data/area grid index", areaGridIndex, hdf5FlatCreateProps);
        }

        void runQuery(const string &modelsDir, const Rounds &rounds, const Ticks &ticks, const PlayerAtTick & playerAtTick,
                      const csknow::behavior_tree_latent_states::BehaviorTreeLatentStates &behaviorTreeLatentStates);
    };

}

#endif //CSKNOW_INFERENCE_LATENT_AREA_H
