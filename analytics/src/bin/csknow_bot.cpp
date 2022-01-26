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
    // \033[A moves up 1 line, \r moves cursor to start of line, \eK clears line
    string upAndClear = "\033[A\r\eK";
    bool needToClearError = false;
    state.numLines = 0;

    while (true) {
        auto start = std::chrono::system_clock::now();
        state.loadServerState(dataPath);
            
        if (!firstFrame) {
            // this handles bot time line
            std::cout << upAndClear;

            // this handles bot inputs
            for (int i = 0; i < state.numLines; i++) {
                std::cout << upAndClear;
            }
        }
        // this handles clearing error from last time
        if (needToClearError) {
            std::cout << upAndClear;
        }
        needToClearError = false;
        if (state.loadedSuccessfully) {
            state.saveBotInputs(dataPath);
            std::cout << state.inputsCopy << std::endl;            
        }
        else {
            needToClearError = true;
            std::cout << "Failed to load " << state.badPath << std::endl;
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> botTime = end - start;
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
