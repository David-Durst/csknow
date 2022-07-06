#include "load_save_bot_data.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/script.h"
#include "bots/testing/scripts/basic_nav.h"
#include "bots/testing/scripts/basic_aim.h"
#include "bots/testing/scripts/basic_memory.h"
#include "bots/testing/scripts/basic_communication.h"
#include "bots/testing/scripts/teamwork.h"
#include "navmesh/nav_file.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <list>
#include <map>
#include <filesystem>
//#define LOG_STATE

int main(int argc, char * argv[]) {
    if (argc != 4) {
        std::cout << "please call this code with 3 arguments: \n"
            << "1. path/to/maps\n"
            << "2. path/to/data\n"
            << "3. path/to/log\n" << std::endl;
        return 1;
    }
    string mapsPath = argv[1], dataPath = argv[2], logPath = argv[3];

    ServerState state;
    state.dataPath = dataPath;

    uint64_t numFailures = 0;
    vector<int> x;
    Tree tree;
    bool finishedTests = false;
    ScriptsRunner scriptsRunner(Script::makeList(
                                            //make_unique<GooseToCatScript>(state),
                                            //make_unique<GooseToCatShortScript>(state),
                                            make_unique<AimAndKillWithinTimeCheck>(state)
                                            //make_unique<PushBaitGooseToCatScript>(state),
                                            //make_unique<PushMultipleBaitGooseToCatScript>(state)
                                            //make_unique<MemoryAimCheck>(state)
                                            //make_unique<MemoryForgetCheck>(state),
                                            //make_unique<CommunicationAimCheck>(state)
                    ), true);


    int32_t priorFrame;
    auto priorStart = std::chrono::system_clock::now();
#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
    while (!finishedTests) {
        auto start = std::chrono::system_clock::now();
        state.loadServerState();
        std::chrono::duration<double> timePerTick(state.loadedSuccessfully ? state.tickInterval : 0.1);
        auto parseEnd = std::chrono::system_clock::now();
        std::chrono::duration<double> startToStart = start - priorStart;
        double frameDiff = (state.getLastFrame() - priorFrame) / 128.;
        priorStart = start;
        priorFrame = state.getLastFrame();
            
        if (state.loadedSuccessfully) {
            tree.tick(state, mapsPath);
            if (state.clients.size() > 0) {
                scriptsRunner.initialize(tree, state);
                finishedTests = scriptsRunner.tick(tree, state);
            }
            state.saveBotInputs();
        }
        else {
            numFailures++;
        }

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> botTime = end - start;
        std::chrono::duration<double> parseTime = parseEnd - start;

        std::fstream logFile (logPath + "/bt_bot.log", std::fstream::out);
        std::fstream testLogFile (logPath + "/bt_test_bot.log", std::fstream::out);
        logFile << "Num failures " << numFailures << ", last bad path: " << state.badPath << std::endl;
        bool sleep;
        if (botTime < timePerTick) {
            logFile << "Bot compute time: ";
            sleep = true;
        }
        else {
            logFile << "\033[1;31mMissed Bot compute time:\033[0m " ;
            sleep = false;
        }
        logFile << botTime.count() << "s, pct parse " << parseTime.count() / botTime.count()
            << ", frame to time ratio" << frameDiff / startToStart.count() << std::endl;
        logFile << tree.curLog;
        logFile.close();
        testLogFile << scriptsRunner.curLog();
        testLogFile.close();
        if (sleep) {
            std::this_thread::sleep_for(timePerTick - botTime);
        }
    }
#pragma clang diagnostic pop

    return 0;
}
