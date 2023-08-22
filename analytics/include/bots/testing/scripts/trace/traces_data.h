//
// Created by durst on 8/20/23.
//

#ifndef CSKNOW_TRACES_DATA_H
#define CSKNOW_TRACES_DATA_H

#include "bots/analysis/feature_store_team.h"

namespace csknow::tests::trace {

    class TracesData {
    public:
        csknow::feature_store::TeamFeatureStoreResult teamFeatureStoreResult;
        vector<string> demoFile;
        vector<int64_t> roundNumber;
        vector<int64_t> ctBot;
        vector<int64_t> oneBotFeatureStoreIndex;
        vector<int64_t> startIndices;
        vector<int64_t> lengths;
        vector<int64_t> numCTBotsNeeded, numTBotsNeeded;
        vector<vector<int64_t>> ctBotIndexToFeatureStoreIndex, tBotIndexToFeatureStoreIndex;

        TracesData(const string & tracesPathStr);
    };
}

#endif //CSKNOW_TRACES_DATA_H
