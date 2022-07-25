//
// Created by steam on 7/9/22.
//

#include "bots/analysis/load_save_vis_points.h"
#include <filesystem>

void VisPoints::launchVisPointsCommand(const ServerState & state) {
    string visPointsFileName = "vis_points.csv";
    string visPointsFilePath = state.dataPath + "/" + visPointsFileName;
    string tmpVisPointsFileName = "vis_points.csv.tmp.write";
    string tmpVisPointsFilePath = state.dataPath + "/" + tmpVisPointsFileName;

    std::stringstream visPointsStream;

    for (const auto & visPoint : visPoints) {
        visPointsStream << visPoint.center.toCSV() << std::endl;
    }

    std::ofstream fsVisPoints(tmpVisPointsFilePath);
    fsVisPoints << visPointsStream.str();
    fsVisPoints.close();

    std::filesystem::rename(tmpVisPointsFilePath, visPointsFilePath);

    state.saveScript({"sm_queryAllVisPointPairs"});
}

void VisPoints::load(string mapsPath, string mapName) {
    string visValidFileName = mapName + ".vis";
    string visValidFilePath = mapsPath + "/" + visValidFileName;

    std::ifstream fsVisValid(visValidFilePath);

    if (std::filesystem::exists(visValidFilePath)) {
        string visValidBuf;
        size_t i = 0;
        while (getline(fsVisValid, visValidBuf)) {
            if (visValidBuf.size() != visPoints.size()) {
                throw std::runtime_error("wrong number of cols in vis valid file's line " + std::to_string(i));
            }

            visPoints[i].visibleFromCurPoint.reset();
            for (size_t j = 0; j < visValidBuf.size(); j++) {
                if (visValidBuf[j] != 't' && visValidBuf[j] != 'f') {
                    throw std::runtime_error("invalid char " + std::to_string(visValidBuf[j]) +
                        " vis valid file's line " + std::to_string(i) + " col " + std::to_string(j));
                }
                visPoints[i].visibleFromCurPoint[j] = visValidBuf[j] == 't';
            }
            i++;
        }
        if (i != visPoints.size()) {
            throw std::runtime_error("wrong number of rows in vis valid file");
        }
    }
    else {
        throw std::runtime_error("no vis valid file");
    }

    // after loading, max sure to or all together, want matrix to be full so can export any row
    for (size_t i = 0; i < visPoints.size(); i++) {
        // set diagonal to true, can see yourself
        visPoints[i].visibleFromCurPoint[i] = true;
        for (size_t j = 0; j < visPoints.size(); j++) {
            visPoints[i].visibleFromCurPoint[j] =
                    visPoints[i].visibleFromCurPoint[j] | visPoints[j].visibleFromCurPoint[i];
            visPoints[j].visibleFromCurPoint[i] = visPoints[i].visibleFromCurPoint[j];
        }
    }
}

