//
// Created by durst on 1/25/22.
//

#ifndef CSKNOW_THINKER_H
#define CSKNOW_THINKER_H
#define SECONDS_BETWEEN_POLICY_CHANGES 30.

#include "load_save_bot_data.h"
#include "bots/input_bits.h"
#include "navmesh/nav_file.h"
#include <chrono>
#include <random>

class Thinker {
    enum class PolicyStates {
        Push,
        Hold
    };

    struct Target {
        int32_t id;
        double distance;
    };

    int curBot, buttonsLastFrame;
    Vec2 lastDeltaAngles;
    ServerState & state;
    nav_mesh::nav_file navFile;
    std::chrono::time_point<std::chrono::system_clock> lastPolicyThinkTime;
    PolicyStates curPolicy;

    void updatePolicy(const ServerState::Client & curClient);
    Target selectTarget(const ServerState::Client & curClient);
    void aimAt(ServerState::Client & curClient, const ServerState::Client & targetClient);
    void fire(ServerState::Client & curClient, const ServerState::Client & targetClient);

    void setButton(ServerState::Client & curClient, int32_t button, bool setTrue) {
        if (setTrue) {
            curClient.buttons |= button;
        }
        else {
            curClient.buttons &= ~button;
        }
    }

    // randomness state https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen; // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis;


public:
    Thinker(ServerState & state, int curBot, string navPath) 
        : state(state), curBot(curBot), lastDeltaAngles{0,0}, navFile(navPath.c_str()),
        lastPolicyThinkTime(std::chrono::system_clock::now()), gen(rd()), dis(0., 1.) {};
    void think();
};


#endif //CSKNOW_THINKER_H

