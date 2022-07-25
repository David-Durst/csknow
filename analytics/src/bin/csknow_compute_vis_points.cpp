#include "bots/load_save_bot_data.h"
#include "bots/behavior_tree/tree.h"
#include "bots/analysis/load_save_vis_points.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <list>
#include <map>
#include <filesystem>
//#define LOG_STATE

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

    VisPoints visPoints(tree.blackboard->navFile);
    visPoints.launchVisPointsCommand(state);

    return 0;
}
