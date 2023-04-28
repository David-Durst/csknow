#include "bots/load_save_bot_data.h"
#include "bots/behavior_tree/tree.h"
#include "navmesh/nav_file.h"
#include "bots/analysis/learned_models.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <list>
#include <map>
#include <filesystem>

//#define LOG_STATE
int main(int argc, char * argv[]) {
    if (argc != 6 && argc != 7) {
        std::cout << "please call this code with 5 or 6 arguments: \n"
            << "1. path/to/maps\n"
            << "2. path/to/data\n"
            << "3. path/to/log\n"
            << "4. path/to/models (string none disables models) \n"
            << "5. r for rounds with models, rh for rounds with hueristics, rht for rounds with t hueristics, rhct for rounds with ct heuristics\n"
            << "6. any value disables read filter names thread\n" << std::endl;
        return 1;
    }
    string mapsPath = argv[1], dataPath = argv[2], logPath = argv[3], modelsDir = argv[4], modelArg = argv[5];

    processModelArg(modelArg);

    ServerState state;
    state.dataPath = dataPath;

    uint64_t numFailures = 0;
    Tree tree(modelsDir);
    std::thread filterReceiver;
    if (argc == 6) {
        std::thread tmpThread(&Tree::readFilterNames, &tree);
        filterReceiver.swap(tmpThread);
    }

    at::set_num_threads(1);

    int32_t priorFrame = 0;
    size_t numMisses = 0;
    size_t numSkips = 0;
    size_t numDups = 0;
    auto priorStart = std::chrono::system_clock::now();
    double savedTickInterval = 0.1;
#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
    while (true) {
        auto start = std::chrono::system_clock::now();
        state.loadServerState();
        std::chrono::duration<double> timePerTick(state.loadedSuccessfully ? state.tickInterval : savedTickInterval);
        savedTickInterval = state.tickInterval;
        auto parseEnd = std::chrono::system_clock::now();
        std::chrono::duration<double> startToStart = start - priorStart;
        double frameDiff = (state.getLastFrame() - priorFrame) / 128.;

        if (state.loadedSuccessfully) {
            tree.tick(state, mapsPath);
            state.saveBotInputs();
        }
        else {
            numFailures++;
        }

        if (state.getLastFrame() - priorFrame > 2) {
            numSkips++;
        }
        if (state.getLastFrame() == priorFrame) {
            numDups++;
        }
        priorStart = start;
        priorFrame = state.getLastFrame();

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> botTime = end - start;
        std::chrono::duration<double> parseTime = parseEnd - start;

        std::fstream logFile (logPath + "/bt_bot.log", std::fstream::out);
        logFile << "Num failures " << numFailures << ", last bad path: " << state.badPath << std::endl;
        bool sleep;
        if (botTime < timePerTick) {
            logFile << "Bot compute time: ";
            sleep = true;
        }
        else {
            logFile << "\033[1;31mMissed Bot compute time:\033[0m " ;
            sleep = false;
            numMisses++;
        }
        logFile << botTime.count() << "s, pct parse " << parseTime.count() / botTime.count()
                << ", start to start " <<  startToStart.count()
                << ", frame to time ratio " << frameDiff / startToStart.count()
                << ", num misses " << numMisses
                << ", num skips " << numSkips
                << ", num dups " << numDups << std::endl;
        logFile << tree.curLog;
        logFile.close();
        if (sleep) {
            std::this_thread::sleep_for(timePerTick - botTime);
        }
    }
#pragma clang diagnostic pop

    exit(0);
}
