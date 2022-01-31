#include "load_save_bot_data.h"
#include "bots/thinker.h"
#include "navmesh/nav_file.h"
#include <iostream>
#include <thread>
#include <chrono>

int main(int argc, char * argv[]) {
    if (argc != 4) {
        std::cout << "please call this code 2 arguments: \n" 
            << "1. path/to/maps\n"
            << "2. path/to/data\n"
            << "3. server_tick_rate" << std::endl;
        return 1;
    }
    string mapsPath = argv[1], dataPath = argv[2];
    string navPath = mapsPath + "/de_dust2.nav";
    int tickRate = std::stoi(argv[3]);
    std::chrono::duration<double> timePerTick(1.0 / tickRate);


    ServerState state;
    Thinker thinker(state, 3, navPath);
    bool firstFrame = true;
    // \033[A moves up 1 line, \r moves cursor to start of line, \33[2K clears line
    string upAndClear = "\033[A\r\33[2K";
    uint64_t numFailures = 0;
    state.numInputLines = 0;

    while (true) {
        auto start = std::chrono::system_clock::now();
        state.loadServerState(dataPath);
            
        if (!firstFrame) {
            // this handles bot time line
            std::cout << upAndClear;

            // this handles the count of failures
            std::cout << upAndClear;

            // this handles bot inputs
            for (int i = 0; i < state.numInputLines; i++) {
                std::cout << upAndClear;
            }
        }
        if (state.loadedSuccessfully) {
            thinker.think();
            state.saveBotInputs(dataPath);
            std::cout << state.inputsCopy << std::endl;            
        }
        else {
            state.numInputLines = 0;
            numFailures++;
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> botTime = end - start;
        std::cout << "Num failures " << numFailures << ", last bad path: " << state.badPath << std::endl;
        if (botTime < timePerTick) {
            std::cout << "Bot compute time: " << botTime.count() << "s" << std::endl;
            std::this_thread::sleep_for(timePerTick - botTime);
        }
        else {
            std::cout << "\033[1;31mMissed Bot compute time:\033[0m " << botTime.count() << "s" << std::endl;
        }
        firstFrame = false;
    }

    return 0;
}
