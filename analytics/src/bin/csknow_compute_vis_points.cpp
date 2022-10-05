#include "bots/load_save_bot_data.h"
#include "bots/behavior_tree/tree.h"
#include "bots/analysis/load_save_vis_points.h"
#include "file_helpers.h"
#include <iostream>

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

    bool ready = true;
    size_t curStartPoint = 0;
    VisPoints visPoints(tree.blackboard->navFile);
    constexpr size_t RAYS_PER_CELL = MAX_NAV_CELLS;
    constexpr size_t RAYS_PER_ITERATION = 750000;
    constexpr size_t CELLS_PER_ITERATION = RAYS_PER_ITERATION / RAYS_PER_CELL;
    VisCommandRange range{0, CELLS_PER_ITERATION};
    bool area = false;
    size_t pointsSize = area ? visPoints.getVisPoints().size() : visPoints.getCellVisPoints().size();
    while (range.startRow < pointsSize) {
        auto start = std::chrono::system_clock::now();
        std::chrono::duration<double> timePerTick(0.5);

        if (ready) {
            visPoints.launchVisPointsCommand(state, area, range);
            ready = false;
        }
        else if (visPoints.readVisPointsCommandResult(state, area, range)) {
            range.startRow += range.numRows;
            if (range.startRow + range.numRows > pointsSize) {
                range.numRows = pointsSize - range.startRow;
            }
            ready = true;
            printProgress(curStartPoint, visPoints.getCellVisPoints().size());
        }

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> botTime = end - start;
        bool sleep = botTime < timePerTick;
        if (sleep) {
            std::this_thread::sleep_for(timePerTick - botTime);
        }
    }

    return 0;
}
