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
    enum class MovementType {
        Push,
        Random,
        Hold,
        NUM_POLICIES
    };

    int movementTypeAsInt(MovementType policy) {
        return static_cast<std::underlying_type_t<MovementType>>(policy);
    }

    struct Target {
        int32_t id;
        double distance;
    };

    struct Plan {
        MovementType policy;
    };

    // constant values across game
    int curBot;
    nav_mesh::nav_file navFile;
    ServerState & state; // technically not constant, but pointer is constant
    bool mustPush;
    ServerState::Client invalidClient;

    // randomness state https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen; // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis;

    // across policy state
    std::chrono::time_point<std::chrono::system_clock> lastPlanTime;
    int32_t lastPlanRound;
    MovementType curMovementType;
    std::vector<nav_mesh::vec3_t> waypoints;
    bool randomLeft, randomRight, randomForward, randomBack;

    // last frame state
    int buttonsLastFrame;
    Vec2 lastDeltaAngles;
    Vec3 lastPushPosition; 

    // cur frame state
    uint64_t curWaypoint;
    bool inSpray;

    Target selectTarget(const ServerState::Client & curClient);
    void updateMovementType(const ServerState::Client & curClient, const ServerState::Client & targetClient);
    void aimAt(ServerState::Client & curClient, const ServerState::Client & targetClient);
    void fire(ServerState::Client & curClient, const ServerState::Client & targetClient);
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

public:
    Thinker(ServerState & state, int curBot, string navPath, bool mustPush) 
        : state(state), curBot(curBot), lastDeltaAngles{0,0}, navFile(navPath.c_str()),
        // init to 24 hours before now so think on first tick
        lastPlanTime(std::chrono::system_clock::now() - std::chrono::hours(24)), 
        lastPlanRound(-1), mustPush(mustPush),
        gen(rd()), dis(0., 1.) {
            invalidClient.serverId = INVALID_SERVER_ID;
        };
    void think();
};


#endif //CSKNOW_THINKER_H

