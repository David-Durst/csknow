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

#include <filesystem>
namespace fs = std::filesystem;

using std::map;
using std::string;
using std::reference_wrapper;

int main(int argc, char * argv[]) {
    if (argc != 3) {
        std::cout << "please call this code 2 arguments: " << std::endl;
        std::cout << "1. path/to/generated_traces" << std::endl;
        std::cout << "2. path/to/baseline_traces" << std::endl;
        return 1;
    }

    string generatedPath = argv[1];
    string baselinePath = argv[2];

    csknow::feature_store::TeamFeatureStoreResult generatedTraces(1, {}), baselineTraces(1, {});
    generatedTraces.load(generatedPath);
    baselineTraces.load(baselinePath);
}