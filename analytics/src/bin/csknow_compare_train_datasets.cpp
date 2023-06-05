//
// Created by durst on 6/2/23.
//
#include <iostream>
#include <unistd.h>
#include <map>
#include <string>
#include <sstream>
#include <functional>
#include <fstream>
#include <iomanip>
#include <ctime>
#include "bots/analysis/feature_store_team.h"
#include "queries/moments/multi_trajectory_similarity.h"

#include <filesystem>
namespace fs = std::filesystem;

using std::map;
using std::string;
using std::reference_wrapper;

int main(int argc, char * argv[]) {
    if (argc != 3) {
        std::cout << "please call this code 2 arguments: " << std::endl;
        std::cout << "1. path/to/predicted_traces" << std::endl;
        std::cout << "2. path/to/groundTruth_traces" << std::endl;
        return 1;
    }

    string predictedPath = argv[1];
    string groundTruthPath = argv[2];

    csknow::feature_store::TeamFeatureStoreResult predictedTraces(1, {}), groundTruthTraces(1, {});
    predictedTraces.load(predictedPath);
    groundTruthTraces.load(groundTruthPath);

    vector<csknow::multi_trajectory_similarity::MultiTrajectorySimilarityResult> multiTrajectorySimilarityResults =
            csknow::multi_trajectory_similarity::computeMultiTrajectorySimilarityForAllPredicted(predictedTraces,
                                                                                                 groundTruthTraces);
}