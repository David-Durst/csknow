#include "load_save_bot_data.h"
#include "bots/thinker.h"
#include "bots/manage_thinkers.h"
#include "navmesh/nav_file.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <list>
#include <map>
#include <filesystem>
//#define LOG_STATE

int main(int argc, char * argv[]) {
    if (argc != 3 && argc != 4) {
        std::cout << "please call this code with 2 or 3 arguments: \n"
            << "1. path/to/maps\n"
            << "2. path/to/data\n"
            << "3. use_model (optional)\n" << std::endl;
        return 1;
    }
    string mapsPath = argv[1], dataPath = argv[2];
    bool useLearned = argc == 4;

    ServerState state;
    //Thinker thinker(state, 3, navPath, true);
    std::list<Thinker> thinkers;

    bool firstFrame = true;
    // \033[A moves up 1 line, \r moves cursor to start of line, \33[2K clears line
    string upAndClear = "\033[A\r\33[2K";
    uint64_t numFailures = 0;
    state.numInputLines = 0;
    state.numThinkLines = 0;
    ManageThinkerState manageThinkerState(dataPath);

    while (true) {
        auto start = std::chrono::system_clock::now();
        state.loadServerState(dataPath);
        std::chrono::duration<double> timePerTick(state.loadedSuccessfully ? state.tickInterval : 0.1);
        auto parseEnd = std::chrono::system_clock::now();
            
        if (!firstFrame) {
#ifdef LOG_STATE
            // this handles bot time line
            std::cout << upAndClear;

            // this handles the count of failures
            std::cout << upAndClear;

            // this handles bot inputs
            for (int i = 0; i < state.numInputLines + state.numThinkLines; i++) {
                std::cout << upAndClear;
            }
#endif // LOG_STATE
            state.inputsLog = "";
            state.thinkLog = "";
            state.numInputLines = 0;
            state.numThinkLines = 0;
        }
        if (state.loadedSuccessfully) {
            manageThinkerState.updateThinkers(state, mapsPath, thinkers, useLearned);
            for (auto & thinker : thinkers) {
                thinker.think();
            }
            state.saveBotInputs(dataPath);
#ifdef LOG_STATE
            std::cout << state.inputsLog << state.thinkLog << std::endl;
#endif // LOG_STATE
        }
        else {
            numFailures++;
        }

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> botTime = end - start;
        std::chrono::duration<double> parseTime = parseEnd - start;
#ifdef LOG_STATE
        std::cout << "Num failures " << numFailures << ", last bad path: " << state.badPath << std::endl;
#endif // LOG_STATE
        if (botTime < timePerTick) {
#ifdef LOG_STATE
            std::cout << "Bot compute time: " << botTime.count()
                << "s, pct parse " << parseTime.count() / botTime.count() << std::endl;
#endif // LOG_STATE
            std::this_thread::sleep_for(timePerTick - botTime);
        }
        else {
#ifdef LOG_STATE
            std::cout << "\033[1;31mMissed Bot compute time:\033[0m " << botTime.count()
                << "s, pct parse " << parseTime.count() / botTime.count() << std::endl;
#endif // LOG_STATE
        }
        firstFrame = false;
    }

    return 0;
}
