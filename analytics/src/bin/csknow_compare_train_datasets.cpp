//
// Created by durst on 6/2/23.
//
#include <iostream>
#include <atomic>
#include <map>
#include <functional>
#include "bots/analysis/feature_store_team.h"
#include "queries/moments/multi_trajectory_similarity.h"
#include "file_helpers.h"

#include <filesystem>
namespace fs = std::filesystem;

using std::map;
using std::string;
using std::reference_wrapper;

void loadTraces(vector<csknow::feature_store::TeamFeatureStoreResult> & traces, const string & pathStr) {
    if (pathStr.find(".hdf5") != string::npos) {
        traces.emplace_back();
        traces.back().load(pathStr, false);
    }
    else {
        vector<fs::path> hdf5Paths;
        for (const auto & entry : fs::directory_iterator(pathStr)) {
            string filename = entry.path().filename();
            if (filename.find(".hdf5") != string::npos && filename.find("behaviorTreeTeamFeatureStore") != string::npos) {
                hdf5Paths.push_back(entry.path());
            }
        }

        traces.resize(hdf5Paths.size());
        std::atomic<size_t> tracesLoaded = 0;
        std::cout << "loading files" << std::endl;
#pragma omp parallel for
        for (size_t i = 0; i < hdf5Paths.size(); i++) {
            traces[i].load(hdf5Paths[i], false);
            tracesLoaded++;
            printProgress(tracesLoaded, traces.size());
        }
        std::cout << std::endl;
    }
}

int main(int argc, char * argv[]) {
    if (argc != 6) {
        std::cout << "please call this code 5 arguments: " << std::endl;
        std::cout << "1. path/to/predicted_traces" << std::endl;
        std::cout << "2. path/to/groundTruth_traces" << std::endl;
        std::cout << "3. output_name" << std::endl;
        std::cout << "4. predicted good rounds - 0 if all round ids, 1 if bot good round ids, 2 if small human good round ids" << std::endl;
        std::cout << "6. ground truth good rounds - 0 if all round ids, 1 if bot good round ids, 2 if small human good round ids" << std::endl;
        return 1;
    }

    string predictedPathStr = argv[1];
    string groundTruthPathStr = argv[2];
    string outputName = argv[3];
    int predictedGoodRoundIdsType = std::stoi(argv[4]), groundTruthGoodRoundIdsType = std::stoi(argv[5]);

    set<int64_t> botGoodRounds{4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 78, 80, 82, 85, 87, 89, 91, 93, 95, 97, 99, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 391, 393, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 490, 492, 494, 496, 498, 500, 502, 504, 506, 508, 510, 512, 514, 516, 518, 520, 522, 524, 526, 528, 530, 532, 534, 537, 539};
    set<int64_t> humanGoodRounds{512, 1, 4, 517, 520, 10, 15, 529, 534, 535, 25, 26, 27, 28, 541, 542, 544, 549, 38, 550, 552, 41, 42, 46, 558, 49, 566, 567, 56, 571, 61, 64, 65, 577, 67, 581, 71, 583, 584, 586, 587, 589, 592, 85, 88, 90, 91, 603, 96, 609, 610, 99, 101, 613, 615, 110, 111, 113, 626, 116, 629, 118, 122, 123, 124, 638, 127, 129, 641, 131, 133, 134, 135, 136, 137, 647, 139, 140, 652, 655, 144,     656, 148, 153, 666, 158, 159, 670, 162, 165, 166, 167, 678, 176, 691, 181, 182, 185, 699, 190, 706, 709, 199, 205, 207, 210, 217, 227, 234, 236, 237, 239, 242, 253, 255, 258, 261, 264, 269, 270, 276, 278, 280, 281, 299, 302, 303, 306, 308, 313, 321, 324, 326, 331, 335, 337, 345, 346, 347, 349, 355, 358, 365, 373, 375, 377, 380, 384, 394, 398, 408, 412, 422, 424, 427, 429, 431, 435, 439, 441, 442, 443, 451, 453, 456, 458, 461, 468, 479, 481, 484, 485, 488, 500, 507, 508};

    vector<csknow::feature_store::TeamFeatureStoreResult> predictedTraces, groundTruthTraces;
    loadTraces(predictedTraces, predictedPathStr);
    if (predictedPathStr != groundTruthPathStr) {
        loadTraces(groundTruthTraces, groundTruthPathStr);
    }
    fs::path logPath = predictedPathStr;
    if (!is_directory(logPath)) {
        logPath = logPath.parent_path();
    }

    auto similarityStart = std::chrono::system_clock::now();
    std::optional<reference_wrapper<const set<int64_t>>> predictedGoodRoundIds, groundTruthGoodRoundIds;
    if (predictedGoodRoundIdsType == 1) {
        predictedGoodRoundIds = botGoodRounds;
    }
    else if (predictedGoodRoundIdsType == 2) {
        predictedGoodRoundIds = humanGoodRounds;
    }
    if (groundTruthGoodRoundIdsType == 1) {
        groundTruthGoodRoundIds = botGoodRounds;
    }
    else if (groundTruthGoodRoundIdsType == 2) {
        groundTruthGoodRoundIds = humanGoodRounds;
    }
    std::cout << "processing similarity" << std::endl;
    csknow::multi_trajectory_similarity::TraceSimilarityResult
    traceSimilarityResult(predictedTraces, predictedPathStr == groundTruthPathStr ? predictedTraces : groundTruthTraces,
                          predictedGoodRoundIds, groundTruthGoodRoundIds, logPath);

    fs::path predictedPath(predictedPathStr);
    fs::path groundTruthPath(groundTruthPathStr);
    fs::path similarityResult;
    // write in smaller data set's folder, default to predicted path if both are directory or file
    if (is_directory(predictedPath) && !is_directory(groundTruthPath)) {
        similarityResult = groundTruthPath.parent_path() / outputName;
    }
    else if (is_directory(predictedPath)) {
        similarityResult = predictedPath / outputName;
    }
    else {
        similarityResult = predictedPath.parent_path() / outputName;
    }
    traceSimilarityResult.toHDF5(similarityResult);
    auto similarityEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> similarityTime = similarityEnd - similarityStart;
    std::cout << "similarity time " << similarityTime.count() << std::endl;
}