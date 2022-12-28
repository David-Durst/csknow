#include "bots/load_save_bot_data.h"
#include "bots/behavior_tree/tree.h"
#include "navmesh/nav_file.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <list>
#include <map>
#include <filesystem>

//#define LOG_STATE
int main(int argc, char * argv[]) {
    if (argc != 4 && argc != 5) {
        std::cout << "please call this code with 3 or 4 arguments: \n"
            << "1. path/to/maps\n"
            << "2. path/to/data\n"
            << "3. path/to/log\n"
            << "4. any value disables read filter names thread\n" << std::endl;
        return 1;
    }
    string mapsPath = argv[1], dataPath = argv[2], logPath = argv[3];

    ServerState state;
    state.dataPath = dataPath;

    uint64_t numFailures = 0;
    Tree tree;
    std::thread filterReceiver;
    if (argc == 4) {
        std::thread tmpThread(&Tree::readFilterNames, &tree);
        filterReceiver.swap(tmpThread);
    }

    at::set_num_threads(1);

#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
    while (true) {
        auto start = std::chrono::system_clock::now();
        state.loadServerState();
        std::chrono::duration<double> timePerTick(state.loadedSuccessfully ? state.tickInterval : 0.1);
        auto parseEnd = std::chrono::system_clock::now();
            
        if (state.loadedSuccessfully) {
            tree.tick(state, mapsPath);
            state.saveBotInputs();
        }
        else {
            numFailures++;
        }

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> botTime = end - start;
        std::chrono::duration<double> parseTime = parseEnd - start;

        std::fstream logFile (logPath + "/bt_bot.log", std::fstream::out);
        logFile << "Num failures " << numFailures << ", last bad path: " << state.badPath << std::endl;
        bool sleep;
        if (botTime < timePerTick) {
            logFile << "Bot compute time: " << botTime.count()
                << "s, pct parse " << parseTime.count() / botTime.count() << std::endl;
            sleep = true;
        }
        else {
            logFile << "\033[1;31mMissed Bot compute time:\033[0m " << botTime.count()
                << "s, pct parse " << parseTime.count() / botTime.count() << std::endl;
            sleep = false;
        }
        logFile << tree.curLog;
        logFile.close();
        if (sleep) {
            std::this_thread::sleep_for(timePerTick - botTime);
        }
    }
#pragma clang diagnostic pop

    return 0;
}
