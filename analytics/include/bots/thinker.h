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

    Vec2 aimAt(int targetClient);

public:
    Thinker(ServerState & state, int curBot) : state(state), curBot(curBot), lastDeltaAngles{0,0} {};
    void think();
};


#endif //CSKNOW_THINKER_H

