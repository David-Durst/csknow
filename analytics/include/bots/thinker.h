// Created by durst on 1/25/22.
//

#ifndef CSKNOW_THINKER_H
#define CSKNOW_THINKER_H
#define SECONDS_BETWEEN_PLAN_CHANGES std::chrono::milliseconds(250)
#define HISTORY_LENGTH 10
#define RETREAT_LENGTH 10
#define MAX_VELOCITY_WHEN_STOPPED 5.
#define HEAD_ADJUSTMENT 8.

#include "load_save_bot_data.h"
#include "bots/input_bits.h"
#include "navmesh/nav_file.h"
#include "geometryNavConversions.h"
#include "circular_buffer.h"
#include "bots/python_plan_model.h"
#include <chrono>
#include <random>
#include <type_traits>
#include <mutex>
#include <thread>
#include <atomic>
#include <array>
#include <csignal>

enum class MovementType {
    Push,
    Retreat,
    Random,
    Hold,
    NUM_TYPES
};

enum class MovementPolicy {
    Normal,
    PushOnly,
    PushAndRetreat,
    HoldOnly,
    NUM_POLICIES
};

template <class T>
constexpr int enumAsInt(T enumElem) {
    return static_cast<std::underlying_type_t<T>>(enumElem);
}

struct Skill {
        bool learned;
        double maxInaccuracy;
        bool stopToShoot;
        MovementPolicy movementPolicy;
};

class Thinker {

    struct Target {
        int32_t csknowId;
        Vec2 offset;
        bool visible;
    };

    class Plan {
    public:
        bool valid = false;
        MovementType movementType = MovementType::Hold;
        Target target{INVALID_ID, {0., 0.}};
        // represents server state for from i and 
        // resulting inputs derived from state i for state i+1
        CircularBuffer<ServerState> stateHistory;
        // time when computing plan started
        std::chrono::time_point<std::chrono::system_clock> computeStartTime, computeEndTime;
        // navigation data for plan
        std::vector<nav_mesh::vec3_t> waypoints;
        uint64_t curWaypoint = -1;
        // save the waypoint from the last plan
        // need to do this at copy time so you get the most recent waypoint
        bool saveWaypoint = false;
        bool randomLeft, randomRight, randomForward, randomBack;
        // track how many times aimed at same target so can get more accurate over time
        uint64_t numTimesRetargeted = 0;

        Plan() : stateHistory(HISTORY_LENGTH) { };

        // for logging infrastructure
        string log;
        int numLogLines = 0;
    };

    // learned planning
    PythonPlanModel planModel;

    // constant values across game
    int32_t curBotCSGOId;
    nav_mesh::nav_file navFile;
    ServerState & liveState; 
    ServerState::Client invalidClient;
    Skill skill;


    // logging for entire thinker
    string thinkLog;
    int numThinkLines = 0;

    // randomness state https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 movementGen, aimGen; // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> movementDis, aimDis;

    // stateForNextPlan builds up the stateHistory buffer for the next plan to make a decision based on
    // developingPlan is the working space for the long term planning
    // executingPLan is the plan to follow for the short term thinking
    Plan stateForNextPlan, developingPlan, executingPlan;
    std::mutex planLock;
    std::thread planThread;
    bool launchedPlanThread = false;
    std::atomic<bool> continuePlanning = true;

    // state across individual frames, shorter term than a plan
    Vec2 lastDeltaAngles;
    bool inSpray;

    // state for planning only
    // locations you've been in past that you can go to again
    CircularBuffer<Vec3> retreatOptions;

    // plan sets long term goals, runs independently of short term thinking
    void plan();
    void selectTarget(ServerState & state);
    void updateMovementType(ServerState curState, ServerState lastState,
                            ServerState oldState, const ServerState::Client & targetClient);

    // short term thinking based on plan
    void aimAt(ServerState::Client & curClient, const ServerState::Client & targetClient,
            const Vec2 & aimOffset, const ServerState::Client & priorClient);
    void fire(ServerState::Client & curClient, const ServerState::Client & targetClient,
            const ServerState::Client & priorClient);
    void moveInDir(ServerState::Client & curClient, Vec2 dir);
    void move(ServerState::Client & curClient, const ServerState::Client & priorClient);
    void defuse(ServerState::Client & curClient, const ServerState::Client & targetClient);

    // helper functions
    ServerState::Client & getCurClient(ServerState & state) {
        int csknowId = state.csgoIdToCSKnowId[curBotCSGOId];
        return state.clients[csknowId];
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

    // helper functions used in planning
    void updateDevelopingPlanWaypoints(const Vec3 & curPosition, const Vec3 & targetPosition);

public:
    Thinker(ServerState & state, int curBotCSGOId, string navPath, Skill skill) 
        : liveState(state), curBotCSGOId(curBotCSGOId), lastDeltaAngles{0, 0}, navFile(navPath.c_str()), 
        skill(skill), retreatOptions(RETREAT_LENGTH),
        movementGen(rd()), aimGen(rd()), movementDis(0., 1.), aimDis(-skill.maxInaccuracy, skill.maxInaccuracy),
        planModel(navFile) {
            invalidClient.csgoId = INVALID_ID;
        };

    ~Thinker() {
        continuePlanning = false;
        if (launchedPlanThread) {
            planThread.join();
        }
    }
    void think();
    int32_t getCurBotCSGOId() const { return curBotCSGOId; }
    Skill getSkill() const { return skill; }
};

int run_model(string moduleName, string functionName);

#endif //CSKNOW_THINKER_H

