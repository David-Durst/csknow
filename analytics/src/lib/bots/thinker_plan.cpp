#include "bots/thinker.h"

void Thinker::plan() {
    while (true) {
        planLock.lock();
        // this copy brings the round number and the state history
        developingPlan = stateForNextPlan;
        // ok to have an empty executing plan as thinking will ignore
        // any invalid plan
        executingPlan = developingPlan;
        planLock.unlock();

        developingPlan.computeStartTime = std::chrono::system_clock::now();
        // make sure state history is filled out
        // otherwise wait as will be filled out in a few frames
        if (developingPlan.getCurSize() == developingPlan.maxSize()) {
            // reset the logging
            developingPlan.numLogLines = 0;
            developingPlan.log = "";

            selectTarget();
            updateMovementType();
            developingPlan.valid = true;
        }
        developingPlan.computeEndTime = std::chrono::system_clock::now();

        std::chrono::duration<double> computeTime = developingPlan.computeEndTime - start;
        developingPlan.numLogLines++;
        developingPlan.logStream << "plan compute time: " << computeTime.count() << "\n";

        if (computeTime < SECONDS_BETWEEN_PLAN_CHANGES) {
            std::this_thread::sleep_for(SECONDS_BETWEEN_PLAN_CHANGES - computeTime);
        }

        developingPlan.log = developingPlan.logStream.str();
        
    }
}

void Thinker::selectTarget() {
    // take from front so get most recent positions
    const ServerState & state = developingPlan.fromFront();
    const ServerState::Client & curClient = getCurClient(state);

    // find the nearest, visible enemy
    // if no enemies are visible, just take the nearest one
    int nearestEnemyServerId = INVALID_SERVER_ID;
    double distance = std::numeric_limits<double>::max();
    bool targetVisible = false;
    for (const auto & otherClient : state.clients) {
        if (otherClient.team != curClient.team && otherClient.isAlive) {
            double otherDistance = computeDistance(
                    {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastEyePosZ},
                    {otherClient.lastEyePosX, otherClient.lastEyePosY, otherClient.lastEyePosZ});
            bool otherVisible = state.visibilityClientPairs.find({ 
                    std::min(curClient.serverId, otherClient.serverId), 
                    std::max(curClient.serverId, otherClient.serverId)
                }) != state.visibilityClientPairs.end();
            if (otherDistance < distance || (otherVisible && !targetVisible)) {
                targetVisible = otherVisible;
                nearestEnemyServerId = otherClient.serverId;
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
        developingPlan.target.id = INVALID_ID;
    }
}

void Thinker::updateMovementType() {
    // take from front so get most recent positions
    const ServerState & state = developingPlan.fromFront();
    const ServerState::Client & curClient = getCurClient(state);
    const ServerState::Client & oldClient = getCurClient(developingPlan.fromBack());
    const ServerState::Client & targetClient = developingPlan.target;

    nav_mesh::vec3_t targetPoint;
    if (targetClient.serverId == INVALID_SERVER_ID) {
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
    developingPlan.randomRight = !randomLeft;
    developingPlan.randomForward = dis(gen) > 0.5;
    developingPlan.randomBack = !randomForward;

    if (mustPush || dis(gen) < 0.8) {
        lastPushPosition = curPosition;
        // choose a new path only if no waypoints or target has changed or plan was for old round
        if (executingPlan.waypoints.empty() || executingPlan.waypoints.back() != targetPoint || 
                executingPlan.roundNumber != developingPlan.roundNumber) {
            try {
                developingPlan.waypoints = navFile.find_path(
                        {curPosition.x, curPosition.y, curPosition.z}, 
                        targetPoint);                    
                if (developingPlan.waypoints.back() != targetPoint) {
                    developingPlan.waypoints.push_back(targetPoint);
                }
            }
            // if waypoint finding fails, just walk randomnly
            catch (const std::exception& e) {
                developingPlan.curMovementType = MovementType::Random;
            }
            curWaypoint = 0;
        }
        // otherwise, keep same path
        else {
            developingPlan.waypoints = executingPlan.waypoints;
        }
        developingPlan.curMovementType = MovementType::Push; 
    }
    // walk randomly if stuck in a push but havent moved during window leading to plan decision
    else if (executingPlan.curMovementType == MovementType::Push &&
            computeDistance(curPosition, oldPosition) < 5.) {
        developingPlan.curMovementType = MovementType::Random;
    }
    else {
        developingPlan.curMovementType = MovementType::Hold; 
    }

    // log results
    developingPlan.logStream << "num waypoints: " << developingPlan.waypoints.size() << ", cur movement type " 
        << movementTypeAsInt(developingPlan.curMovementType) << "\n";
    developingPlan.numLogLines++;

    if (developingPlan.waypoints.size() > curWaypoint) {
        developingPlan.logStream << "cur waypoint " << curWaypoint << ":" 
            << developingPlan.waypoints[curWaypoint].x << "," << developingPlan.waypoints[curWaypoint].y 
            << "," << developingPlan.waypoints[curWaypoint].z << "\n";
        developingPlan.numLogLines++;
    }
    if (!developingPlan.waypoints.empty()) {
        uint64_t lastWaypoint = waypoints.size() - 1;
        developingPlan.logStream << "last waypoint " << lastWaypoint << ":" 
            << developingPlan.waypoints[lastWaypoint].x << "," << developingPlan.waypoints[lastWaypoint].y 
            << "," << developingPlan.waypoints[lastWaypoint].z << "\n";
        developingPlan.numLogLines++;
    }
}
