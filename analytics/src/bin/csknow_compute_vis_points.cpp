#include "bots/load_save_bot_data.h"
#include "bots/behavior_tree/tree.h"
#include "bots/analysis/load_save_vis_points.h"
#include "file_helpers.h"
#include <iostream>
#include <cstdlib>

constexpr size_t RAYS_PER_ITERATION = 750000;

void computeVisPoints(const ServerState & state, const Tree & tree, bool area, const string & mapsPath) {
    std::cout << std::endl << "start compute " << (area ? "area" : "cells") << std::endl;
    auto startCompute = std::chrono::system_clock::now();

    bool ready = true;
    VisPoints visPoints(tree.blackboard->navFile);
    VisCommandRange range{0, 0};
    size_t pointsSize = area ? visPoints.getAreaVisPoints().size() : visPoints.getCellVisPoints().size();
    visPoints.clearFiles(state);
    while (range.startRow < pointsSize) {
        auto start = std::chrono::system_clock::now();
        std::chrono::duration<double> timePerTick(0.1);

        if (ready) {
            size_t numRays = 0;
            for (range.numRows = 1; range.startRow + range.numRows < pointsSize; range.numRows++) {
                numRays += pointsSize - (range.startRow + range.numRows);
                if (numRays >= RAYS_PER_ITERATION) {
                    break;
                }
            }
            visPoints.launchVisPointsCommand(state, area, range);
            ready = false;
        }
        else if (visPoints.readVisPointsCommandResult(state, area, range)) {
            range.startRow += range.numRows;
            ready = true;
            if (area) {
                printProgress(range.startRow, visPoints.getAreaVisPoints().size());
            }
            else {
                printProgress(range.startRow, visPoints.getCellVisPoints().size());
            }
        }

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> botTime = end - start;
        bool sleep = botTime < timePerTick;
        if (sleep) {
            std::this_thread::sleep_for(timePerTick - botTime);
        }
    }
    auto endCompute = std::chrono::system_clock::now();
    std::chrono::duration<double> computeTime = endCompute - startCompute;
    std::cout << "end compute " << computeTime.count() << std::endl;

    std::cout << std::endl << "start write" << std::endl;
    auto startWrite = std::chrono::system_clock::now();
    visPoints.save(mapsPath, "de_dust2", area);
    auto endWrite = std::chrono::system_clock::now();
    std::chrono::duration<double> writeTime = endWrite - startWrite;
    std::cout << "end write " << writeTime.count() << std::endl;

    std::cout << std::endl << "start read" << std::endl;
    auto startRead = std::chrono::system_clock::now();
    VisPoints visPointsCompare(tree.blackboard->navFile);
    visPointsCompare.load(mapsPath, "de_dust2", area, tree.blackboard->navFile);
    auto endRead = std::chrono::system_clock::now();
    std::chrono::duration<double> readTime = endRead - startRead;
    std::cout << "end read " << readTime.count() << std::endl;

    if (area) {
        for (size_t i = 0; i < visPoints.getAreaVisPoints().size(); i++) {
            if (visPoints.getAreaVisPoints()[i].visibleFromCurPoint !=
                visPointsCompare.getAreaVisPoints()[i].visibleFromCurPoint) {
                std::cout << "row mismatch: " << i << std::endl;
            }
        }
    }
    else {
        for (size_t i = 0; i < visPoints.getCellVisPoints().size(); i++) {
            if (visPoints.getCellVisPoints()[i].visibleFromCurPoint !=
                visPointsCompare.getCellVisPoints()[i].visibleFromCurPoint) {
                std::cout << "row mismatch: " << i << std::endl;
            }
        }
    }

}

int main(int argc, char * argv[]) {
    if (argc != 3) {
        std::cout << "please call this code with 2 arguments: \n"
            << "1. path/to/maps\n"
            << "2. path/to/data\n" << std::endl;
        return 1;
    }
    string mapsPath = argv[1], dataPath = argv[2];

    ServerState state;
    state.dataPath = dataPath;

    Tree tree;

    state.loadServerState();
    if (state.loadedSuccessfully) {
        tree.tick(state, mapsPath);
        state.saveBotInputs();
    }
    else {
        throw std::runtime_error("failed to load server state");
    }

    computeVisPoints(state, tree, true, mapsPath);
    computeVisPoints(state, tree, false, mapsPath);

    return 0;
}
