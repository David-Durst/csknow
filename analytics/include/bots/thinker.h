//
// Created by durst on 1/25/22.
//

#ifndef CSKNOW_THINKER_H
#define CSKNOW_THINKER_H

#include "load_save_bot_data.h"
#include "bots/input_bits.h"

class Thinker {
    int curBot;
    Vec2 lastDeltaAngles;
    ServerState & state;

    struct Target {
        int32_t id;
        double distance;
    };
    Target selectTarget(const ServerState::Client & curClient);
    void aimAt(ServerState::Client & curClient, const ServerState::Client & targetClient);
    void fire(ServerState::Client & curClient, const ServerState::Client & targetClient);

public:
    Thinker(ServerState & state, int curBot) : state(state), curBot(curBot), lastDeltaAngles{0,0} {};
    void think();
};


#endif //CSKNOW_THINKER_H

