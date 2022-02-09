#include "bots/thinker.h"
#include <string>

void Thinker::plan() {
    while (true) {
        planLock.lock();
        // ok to have an empty executing plan as thinking will ignore
        // any invalid plan
        if (developingPlan.saveWaypoint) {
            developingPlan.curWaypoint = executingPlan.curWaypoint;
        }
        executingPlan = developingPlan;
        // this copy brings the round number and the state history
        developingPlan = stateForNextPlan;
        planLock.unlock();

        developingPlan.computeStartTime = std::chrono::system_clock::now();
        // reset the logging
        developingPlan.numLogLines = 0;
        developingPlan.log = "";

        // make sure state history is filled out
        // otherwise wait as will be filled out in a few frames
        if (developingPlan.stateHistory.getCurSize() == developingPlan.stateHistory.maxSize()) {

            // take from front so get most recent positions
            ServerState & state = developingPlan.stateHistory.fromFront();
            const ServerState::Client & curClient = getCurClient(state);
            // take from back so get oldest positions
            const ServerState::Client & oldClient = getCurClient(developingPlan.stateHistory.fromBack());

            selectTarget(state, curClient);
            const ServerState::Client & targetClient = state.clients[developingPlan.target.csknowId];

            updateMovementType(state, curClient, oldClient, targetClient);

            developingPlan.valid = true;
        }
        developingPlan.computeEndTime = std::chrono::system_clock::now();

        std::chrono::duration<double> computeTime = developingPlan.computeEndTime - developingPlan.computeStartTime;
        developingPlan.numLogLines++;
        developingPlan.log += "plan compute time: " + std::to_string(computeTime.count()) + "\n";

        if (computeTime < SECONDS_BETWEEN_PLAN_CHANGES) {
            std::this_thread::sleep_for(SECONDS_BETWEEN_PLAN_CHANGES - computeTime);
        }
    }
}

void Thinker::selectTarget(const ServerState state, const ServerState::Client & curClient) {
    // find the nearest, visible enemy
    // if no enemies are visible, just take the nearest one
    int nearestEnemyServerId = INVALID_ID;
    double distance = std::numeric_limits<double>::max();
    bool targetVisible = false;
    for (const auto & otherClient : state.clients) {
        if (otherClient.team != curClient.team && otherClient.isAlive) {
            double otherDistance = computeDistance(
                    {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastEyePosZ},
                    {otherClient.lastEyePosX, otherClient.lastEyePosY, otherClient.lastEyePosZ});
            bool otherVisible = state.visibilityClientPairs.find({ 
                    std::min(curClient.csgoId, otherClient.csgoId),
                    std::max(curClient.csgoId, otherClient.csgoId)
                }) != state.visibilityClientPairs.end();
            if (otherDistance < distance || (otherVisible && !targetVisible)) {
                targetVisible = otherVisible;
                nearestEnemyServerId = otherClient.csgoId;
                distance = otherDistance;
            }
        }
    }
    
    // if found any targets, set them, otherwise mark target as invalid
    if (nearestEnemyServerId != INVALID_ID) {
        // TODO: make offset random based on skill, more skill less offset from target
        developingPlan.target = {state.csgoIdToCSKnowId[nearestEnemyServerId], {0, 0, 0}};
    }
    else {
        developingPlan.target.csknowId = INVALID_ID;
    }
}

void Thinker::updateMovementType(const ServerState state, const ServerState::Client & curClient,
        const ServerState::Client & oldClient, const ServerState::Client & targetClient) {
    nav_mesh::vec3_t targetPoint;
    if (targetClient.csgoId == INVALID_ID) {
        targetPoint = {state.c4X, state.c4Y, state.c4Z}; 
    }
    else {
        targetPoint = {targetClient.lastEyePosX, targetClient.lastEyePosY, targetClient.lastFootPosZ};
    }
    
    Vec3 curPosition{curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ};
    Vec3 oldPosition{oldClient.lastEyePosX, oldClient.lastEyePosY, oldClient.lastFootPosZ};

    // set random directions here so that random movement type chosen either due to 
    // nav path failure or getting stuck has new random directions to go in
    developingPlan.randomLeft = dis(gen) > 0.5;
    developingPlan.randomRight = !developingPlan.randomLeft;
    developingPlan.randomForward = dis(gen) > 0.5;
    developingPlan.randomBack = !developingPlan.randomForward;

    developingPlan.saveWaypoint = false;

    if (mustPush || dis(gen) < 0.8) {
        // choose a new path only if no waypoints or target has changed or plan was for old round
        if (executingPlan.waypoints.empty() || executingPlan.waypoints.back() != targetPoint || 
                executingPlan.stateHistory.fromFront().roundNumber != developingPlan.stateHistory.fromFront().roundNumber) {
            try {
                developingPlan.waypoints = navFile.find_path(
                        {static_cast<float>(curPosition.x), static_cast<float>(curPosition.y),
                         static_cast<float>(curPosition.z)},
                        targetPoint);                    
                if (developingPlan.waypoints.back() != targetPoint) {
                    developingPlan.waypoints.push_back(targetPoint);
                }
            }
            // if waypoint finding fails, just walk randomnly
            catch (const std::exception& e) {
                developingPlan.movementType = MovementType::Random;
            }
            developingPlan.curWaypoint = 0;
        }
        // otherwise, keep same path
        else {
            developingPlan.waypoints = executingPlan.waypoints;
            developingPlan.saveWaypoint = true;
        }
        developingPlan.movementType = MovementType::Push;
    }
    // walk randomly if stuck in a push but havent moved during window leading to plan decision
    else if (executingPlan.movementType == MovementType::Push &&
            computeDistance(curPosition, oldPosition) < 5.) {
        developingPlan.movementType = MovementType::Random;
    }
    else {
        developingPlan.movementType = MovementType::Hold;
    }

    // log results
    std::stringstream logStream;
    logStream << "num waypoints: " << developingPlan.waypoints.size() << ", cur movement type " 
        << movementTypeAsInt(developingPlan.movementType) << "\n";
    developingPlan.numLogLines++;

    if (!developingPlan.waypoints.empty()) {
        uint64_t lastWaypoint = developingPlan.waypoints.size() - 1;
        logStream << "last waypoint " << lastWaypoint << ":" 
            << developingPlan.waypoints[lastWaypoint].x << "," << developingPlan.waypoints[lastWaypoint].y 
            << "," << developingPlan.waypoints[lastWaypoint].z << "\n";
        developingPlan.numLogLines++;
    }
    developingPlan.log += logStream.str();
}
