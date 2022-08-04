#include "bots/load_save_bot_data.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/script.h"
#include "bots/testing/scripts/navigation/test_entry.h"
#include "bots/testing/scripts/navigation/test_hold.h"
#include "bots/testing/scripts/test_aim.h"
#include "bots/testing/scripts/test_memory.h"
#include "bots/testing/scripts/test_communication.h"
#include "bots/testing/scripts/test_teamwork.h"
#include "bots/testing/scripts/test_danger.h"
#include "bots/testing/scripts/test_possible_nav_areas.h"
#include "bots/testing/scripts/test_engage_spacing.h"
#include "bots/testing/scripts/test_defuse.h"
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
    std::thread filterReceiver(&Tree::readFilterNames, &tree);

    bool finishedTests = false;
    ScriptsRunner scriptsRunner(Script::makeList(
                                            //make_unique<GooseToCatScript>(state)
                                            //make_unique<GooseToCatShortScript>(state),
                                            //make_unique<CTPushLongScript>(state),
                                            make_unique<DefuseScript>(state)
                                            //make_unique<HoldLongScript>(state),
                                            //make_unique<HoldASitePushScript>(state),
                                            //make_unique<HoldASiteBaitScript>(state),
                                            //make_unique<HoldBSitePushScript>(state),
                                            //make_unique<HoldBSiteBaitScript>(state)
                                            //make_unique<AimAndKillWithinTimeCheck>(state),
                                            //make_unique<CTEngageSpacingScript>(state)
                                            //make_unique<PushBaitGooseToCatScript>(state),
                                            //make_unique<PushWaitForBaitGooseToCatScript>(state),
                                            //make_unique<PushMultipleBaitGooseToCatScript>(state),
                                            //make_unique<PushLurkBaitASiteScript>(state),
                                            //make_unique<MemoryAimCheck>(state),
                                            //make_unique<MemoryForgetCheck>(state),
                                            //make_unique<CommunicationAimCheck>(state),
                                            //make_unique<CommunicationForgetCheck>(state),
                                            //make_unique<CommunicationIgnoreCheck>(state),
                                            //make_unique<SpawnPossibleNavAreasCheck>(state),
                                            //make_unique<DiffusionPossibleNavAreasCheck>(state),
                                            //make_unique<VisibilityPossibleNavAreasCheck>(state),
                                            //make_unique<DangerOnePlayerCheck>(state),
                                            //make_unique<DangerTwoPlayerCheck>(state)
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

        if (state.loadedSuccessfully) {
            tree.tick(state, mapsPath);
            if (state.clients.size() > 0) {
                //std::cout << "time since last save " << state.getSecondsBetweenTimes(start, priorStart) << std::endl;
                scriptsRunner.initialize(tree, state);
                finishedTests = scriptsRunner.tick(tree, state);
            }
            state.saveBotInputs();
        }
        else {
            numFailures++;
        }

        priorStart = start;
        priorFrame = state.getLastFrame();

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
            << ", start to start " <<  startToStart.count()
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
