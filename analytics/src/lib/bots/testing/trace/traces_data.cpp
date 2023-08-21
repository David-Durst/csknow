//
// Created by durst on 8/20/23.
//

#include "bots/testing/scripts/trace/traces_data.h"

namespace csknow::tests::trace {
    TracesData::TracesData(const std::string &tracesPathStr) {
        teamFeatureStoreResult.load(tracesPathStr);

        HighFive::File file(tracesPathStr, HighFive::File::ReadOnly);
        demoFile = file.getDataSet("/extra/demo file").read<std::vector<string>>();
        roundNumber = file.getDataSet("/extra/round number").read<std::vector<int64_t>>();
        startIndices = file.getDataSet("/extra/start index in hdf5").read<std::vector<int64_t>>();
        lengths = file.getDataSet("/extra/length").read<std::vector<int64_t>>();

        numCTBotsNeeded.resize(lengths.size(), 0);
        numTBotsNeeded.resize(lengths.size(), 0);
        ctBotIndexToFeatureStoreIndex.resize(lengths.size(), {});
        tBotIndexToFeatureStoreIndex.resize(lengths.size(), {});

        for (size_t i = 0; i < startIndices.size(); i++) {
            int64_t startIndex = startIndices[i];
            for (size_t columnDataIndex = 0; columnDataIndex < teamFeatureStoreResult.getAllColumnData().size();
                columnDataIndex++) {
                const array<feature_store::TeamFeatureStoreResult::ColumnPlayerData, feature_store::max_enemies> &
                        columnData = teamFeatureStoreResult.getAllColumnData()[columnDataIndex];
                for (size_t columnPlayer = 0; columnPlayer < columnData.size(); columnPlayer++) {
                    if (columnData[columnPlayer].alive[startIndex]) {
                        if (columnData[columnPlayer].ctTeam[startIndex]) {
                            ctBotIndexToFeatureStoreIndex[i].push_back(columnPlayer);
                            numCTBotsNeeded[i]++;
                        }
                        else {
                            tBotIndexToFeatureStoreIndex[i].push_back(columnPlayer);
                            numTBotsNeeded[i]++;
                        }
                    }
                }
            }
        }
    }
}