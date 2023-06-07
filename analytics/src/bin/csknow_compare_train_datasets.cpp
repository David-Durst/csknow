//
// Created by durst on 6/2/23.
//
#include <iostream>
#include <map>
#include <functional>
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

    string predictedPathStr = argv[1];
    string groundTruthPathStr = argv[2];

    csknow::feature_store::TeamFeatureStoreResult predictedTraces(1, {}), groundTruthTraces(1, {});
    predictedTraces.load(predictedPathStr);
    groundTruthTraces.load(groundTruthPathStr);

    csknow::multi_trajectory_similarity::TraceSimilarityResult traceSimilarityResult(predictedTraces, groundTruthTraces);
    fs::path predictedPath(predictedPathStr);
    fs::path similarityResult = predictedPath.parent_path() / "pathSimilarity.hdf5";
    traceSimilarityResult.toHDF5(similarityResult);
}