#include "load_save_bot_data.h"
#include <iostream>
#include <chrono>

int main(int argc, char * argv[]) {
    if (argc != 2) {
        std::cout << "please call this code 1 arguments: \n" 
            << "1. path/to/data" << std::endl;
        return 1;
    }
    string dataPath = argv[1];

    ServerState state;

    while (true) {
        state.loadServerState(dataPath);
        if (loadedSuccessfully) {
            for (int i = 0; i < state.numLines; i++) {
                std::cout << "\r";
            }
            std::cout << state.inputsCopy.str();            

            state.saveBotInputs(dataPath);
        }
        else {
            std::cout << "\rFailed to load state";
        }
    }

    auto 

    return 0;
}
