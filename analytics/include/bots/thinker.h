//
// Created by durst on 1/25/22.
//

#ifndef CSKNOW_THINKER_H
#define CSKNOW_THINKER_H
#define SECONDS_BETWEEN_PLAN_CHANGES std::chrono::seconds(0.5)
#define HISTORY_LENGTH 10

#include "load_save_bot_data.h"
#include "bots/input_bits.h"
#include "navmesh/nav_file.h"
#include "circular_buffer.h"
#include <chrono>
#include <random>
#include <type_traits>
#include <mutex>
#include <thread>

class Thinker {
    enum class MovementType {
        Push,
        Random,
        Hold,
        NUM_POLICIES
    };

    int movementTypeAsInt(MovementType movementType) {
        return static_cast<std::underlying_type_t<MovementType>>(movementType);
    }

    struct Target {
        int32_t id;
        Vec3 offset;
    };

    class Plan {
    public:
        bool valid = false;
        MovementType movementType = MovementType::Hold;
        Target target;
        // represents server state for from i and 
        // resulting inputs derived from state i for state i+1
        CircularBuffer<ServerState> stateHistory;
        // round during which state is collected, shouldn't have history from different rounds
        int32_t roundNumber;
        // time when computing plan started
        std::chrono::time_point<std::chrono::system_clock> computeStartTime, computeEndTime;
        // navigation data for plan
        std::vector<nav_mesh::vec3_t> waypoints;
        uint64_t curWaypoint;
        bool randomLeft, randomRight, randomForward, randomBack;

        Plan() : stateHistory(HISTORY_LENGTH) {
            target.id = INVALID_SERVER_ID;
        };

        // for logging infrastructure
        std::stringstream logStream;
        string log;
        int numLogLines;
    };

    // constant values across game
    int curBot;
    nav_mesh::nav_file navFile;
    ServerState & liveState; 
    bool mustPush;
    ServerState::Client invalidClient;

    // randomness state https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen; // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis;

    // stateForNextPlan builds up the stateHistory buffer for the next plan to make a decision based on
    // developingPlan is the working space for the long term planning
    // executingPLan is the plan to follow for the short term thinking
    Plan stateForNextPlan, developingPlan, executingPlan;
    std::mutex planLock;
    std::thread planThread;
    bool launchedPlanThread = false;

    // state across individual frames, shorter term than a plan
    Vec2 lastDeltaAngles;
    bool inSpray;

    // plan sets long term goals, runs independently of short term thinking
    void plan();
    void selectTarget();
    void updateMovementType();

    // short term thinking based on plan
    void aimAt(ServerState::Client & curClient, const ServerState::Client & targetClient);
    void fire(ServerState::Client & curClient, const ServerState::Client & targetClient);
    void move(ServerState::Client & curClient);
    void defuse(ServerState::Client & curClient, const ServerState::Client & targetClient);

    // helper functions
    ServerState::Client & getCurClient(ServerState & state) {
        int csknowId = state.serverClientIdToCSKnowId[curBot];
        ServerState::Client & curClient = state.clients[csknowId];
    }
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
        : curState(state), curBot(curBot), lastDeltaAngles{0,0}, navFile(navPath.c_str()),
        , mustPush(mustPush),
        gen(rd()), dis(0., 1.) {
            invalidClient.serverId = INVALID_SERVER_ID;
            // init to 24 hours before now so think on first tick
            curPlan.computeEndTime = std::chrono::system_clock::now() - std::chrono::hours(24);
            curplan.computeStartRound = -1;
        };
    void think();
};


#endif //CSKNOW_THINKER_H

