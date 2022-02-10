#include "load_save_bot_data.h"
#include "bots/thinker.h"
#include "navmesh/nav_file.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <list>
#include <map>

void updateThinkers(ServerState & state, string navPath, std::list<Thinker> & thinkers) {
    std::map<int32_t, bool> botCSGOIdsToAdd;
    for (const auto & client : state.clients) {
        if (client.isBot) {
            botCSGOIdsToAdd.insert({client.csgoId, true});
        }
    }

    // remove all elements that aren't currently bots
    thinkers.remove_if([&](const Thinker & thinker){ 
            return botCSGOIdsToAdd.find(thinker.getCurBotCSGOId()) == botCSGOIdsToAdd.end() ; 
        });

    // mark already controlled bots as not needed to be added
    for (const auto & thinker : thinkers) {
        botCSGOIdsToAdd[thinker.getCurBotCSGOId()] = false;
    }

    // add all uncontrolled bots
    for (const auto & [csgoId, toAdd] : botCSGOIdsToAdd) {
        if (toAdd) {
            thinkers.emplace_back(state, csgoId, navPath, true);
        }
    }

    // clear all old inputs
    // and force existing players to put a new one in
    // this will happen infrequrntly, so unlikely to cause issue with frequent missed inputs
    for (size_t i = 0; i < state.inputsValid.size(); i++) {
        state.inputsValid[i] = false;
    }
}

int main(int argc, char * argv[]) {
    if (argc != 4) {
        std::cout << "please call this code 2 arguments: \n" 
            << "1. path/to/maps\n"
            << "2. path/to/data\n"
            << "3. server_tick_rate" << std::endl;
        return 1;
    }
    string mapsPath = argv[1], dataPath = argv[2];
    int tickRate = std::stoi(argv[3]);
    std::chrono::duration<double> timePerTick(1.0 / tickRate);


    ServerState state;
    // load once so can initialize rest of structs with data
    state.loadServerState(dataPath);
    string navPath = mapsPath + "/" + state.mapName + ".nav";
    //Thinker thinker(state, 3, navPath, true);
    std::list<Thinker> thinkers;

    bool firstFrame = true;
    // \033[A moves up 1 line, \r moves cursor to start of line, \33[2K clears line
    string upAndClear = "\033[A\r\33[2K";
    uint64_t numFailures = 0;
    state.numInputLines = 0;
    state.numThinkLines = 0;

    while (true) {
        auto start = std::chrono::system_clock::now();
        state.loadServerState(dataPath);
            
        if (!firstFrame) {
            // this handles bot time line
            std::cout << upAndClear;

            // this handles the count of failures
            std::cout << upAndClear;

            // this handles bot inputs
            for (int i = 0; i < state.numInputLines + state.numThinkLines; i++) {
                std::cout << upAndClear;
            }
            state.inputsLog = "";
            state.thinkLog = "";
            state.numInputLines = 0;
            state.numThinkLines = 0;
        }
        if (state.loadedSuccessfully) {
            updateThinkers(state, navPath, thinkers);
            for (auto & thinker : thinkers) {
                thinker.think();
            }
            //thinker.think();
            state.saveBotInputs(dataPath);
            std::cout << state.inputsLog << state.thinkLog << std::endl;
        }
        else {
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
