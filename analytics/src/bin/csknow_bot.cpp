#include "load_save_bot_data.h"
#include <iostream>
#include <thread>
#include <chrono>

int main(int argc, char * argv[]) {
    if (argc != 3) {
        std::cout << "please call this code 2 arguments: \n" 
            << "1. path/to/data\n"
            << "2. server_tick_rate" << std::endl;
        return 1;
    }
    string dataPath = argv[1];
    int tickRate = std::stoi(argv[2]);
    std::chrono::duration<double> timePerTick(1.0 / tickRate);

    ServerState state;
    bool firstFrame = true;
    string upAndClear = "\33[2K\r";

    while (true) {
        auto start = std::chrono::system_clock::now();
        state.loadServerState(dataPath);
            
        // this handles bot time line
        if (!firstFrame) {
            std::cout << upAndClear;
        }
        if (state.loadedSuccessfully) {
            if (!firstFrame) {
                for (int i = 0; i < state.numLines; i++) {
                    std::cout << upAndClear;
                }
            }
            std::cout << state.inputsCopy.str() << std::endl;            

            state.saveBotInputs(dataPath);
        }
        else {
            // this handles failure line
            if (!firstFrame) {
                std::cout << upAndClear;
            }
            std::cout << "Failed to load state" << std::endl;
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> botTime = end - start;
        std::cout << "Bot compute time: " << botTime.count() << "s" << std::endl;
        if (botTime < timePerTick) {
            std::this_thread::sleep_for(timePerTick - botTime);
        }
    }

    return 0;
}
