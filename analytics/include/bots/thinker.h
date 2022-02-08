//
// Created by durst on 1/25/22.
//

#ifndef CSKNOW_THINKER_H
#define CSKNOW_THINKER_H
#define SECONDS_BETWEEN_POLICY_CHANGES 5.

#include "load_save_bot_data.h"
#include "bots/input_bits.h"
#include "navmesh/nav_file.h"
#include <chrono>
#include <random>
#include <type_traits>

class Thinker {
    enum class PolicyStates {
        Push,
        Random,
        Hold,
        NUM_POLICIES
    };

    int policyAsInt(PolicyStates policy) {
        return static_cast<std::underlying_type_t<PolicyStates>>(policy);
    }

    struct Target {
        int32_t id;
        double distance;
    };

    int curBot, buttonsLastFrame;
    Vec2 lastDeltaAngles;
    ServerState & state;
    nav_mesh::nav_file navFile;
    std::chrono::time_point<std::chrono::system_clock> lastPolicyThinkTime;
    int32_t lastPolicyRound;
    PolicyStates curPolicy;
    bool mustPush;
    std::vector<nav_mesh::vec3_t> waypoints;
    uint64_t curWaypoint;
    bool randomLeft = true, randomRight, randomForward, randomBack;

    Target selectTarget(const ServerState::Client & curClient);
    ServerState::Client invalidClient;
    void updatePolicy(const ServerState::Client & curClient, const ServerState::Client & targetClient);
    Vec3 oldPosition;
    void aimAt(ServerState::Client & curClient, const ServerState::Client & targetClient);
    void fire(ServerState::Client & curClient, const ServerState::Client & targetClient);
    bool inSpray;
    void move(ServerState::Client & curClient);
    void defuse(ServerState::Client & curClient, const ServerState::Client & targetClient);

    void setButton(ServerState::Client & curClient, int32_t button, bool setTrue) {
        if (setTrue) {
            curClient.buttons |= button;
        }
        else {
            curClient.buttons &= ~button;
        }
    }

    bool getButton(ServerState::Client & curClient, int32_t button) {
        return curClient.buttons & button > 0;
    }

    // randomness state https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen; // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis;


public:
    Thinker(ServerState & state, int curBot, string navPath, bool mustPush) 
        : state(state), curBot(curBot), lastDeltaAngles{0,0}, navFile(navPath.c_str()),
        // init to 24 hours before now so think on first tick
        lastPolicyThinkTime(std::chrono::system_clock::now() - std::chrono::hours(24)), 
        lastPolicyRound(-1), mustPush(mustPush),
        gen(rd()), dis(0., 1.) {
            invalidClient.serverId = INVALID_SERVER_ID;
        };
    void think();
};


#endif //CSKNOW_THINKER_H

