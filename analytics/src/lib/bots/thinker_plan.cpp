#include "bots/thinker.h"
#include <string>
#include <functional>

void Thinker::plan() {
    while (continuePlanning) {
        planLock.lock();
        if (executingPlan.valid) {
            // ok to have an empty executing plan as thinking will ignore
            // any invalid plan
            if (developingPlan.saveWaypoint) {
                developingPlan.curWaypoint = executingPlan.curWaypoint;
            }
            // if can match progress in current executing plan to next one, then start at the same point
            // in the new executing plan
            else if (developingPlan.stateHistory.fromFront().roundNumber == executingPlan.stateHistory.fromFront().roundNumber && 
                    executingPlan.curWaypoint > 0) {
                for (size_t i = 0; i < developingPlan.waypoints.size(); i++) {
                    if (executingPlan.waypoints[executingPlan.curWaypoint - 1] == developingPlan.waypoints[i]) {
                        // if on last waypoint in new plan, don't jump over the end
                        developingPlan.curWaypoint = std::min(i+1, developingPlan.waypoints.size() - 1);
                    }
                }
            }
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

            // clear retreats on new round
            if (executingPlan.stateHistory.fromFront().roundNumber != 
                    developingPlan.stateHistory.fromFront().roundNumber) {
                retreatOptions.clear();
            }
            // you can retreat to the current location
            Vec3 curPosition = {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ};
            if (retreatOptions.getCurSize() == 0 || curPosition != retreatOptions.fromFront()) {
                retreatOptions.enqueue(curPosition, true);
            }

            selectTarget(state, curClient);
            const ServerState::Client & targetClient = developingPlan.target.csknowId == INVALID_ID ?
                invalidClient : state.clients[developingPlan.target.csknowId];

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

std::map<int32_t, bool>
getCSGOIdToVisibleClients(const ServerState & state, const ServerState::Client & curClient) {
    std::map<int32_t, bool> result;
    for (const auto & otherClient : state.clients) {
        bool otherVisible = state.visibilityClientPairs.find({ 
                std::min(curClient.csgoId, otherClient.csgoId),
                std::max(curClient.csgoId, otherClient.csgoId)
            }) != state.visibilityClientPairs.end();
        result.insert({otherClient.csgoId, otherVisible});
    }
    return result;
}

void Thinker::selectTarget(const ServerState & state, const ServerState::Client & curClient) {
    // find the nearest, visible enemy
    // if no enemies are visible, just take the nearest one
    int nearestEnemyServerId = INVALID_ID;
    double distance = std::numeric_limits<double>::max();
    bool targetVisible = false;
    std::map<int32_t, bool> csgoIdToVisible = getCSGOIdToVisibleClients(state, curClient);
    // need to explicitly give type so cast reference_wrapper to ref
    for (const ServerState::Client & otherClient : state.clients) {
        if (otherClient.team != curClient.team && otherClient.isAlive) {
            double otherDistance = computeDistance(
                    {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastEyePosZ},
                    {otherClient.lastEyePosX, otherClient.lastEyePosY, otherClient.lastEyePosZ});
            bool otherVisible = csgoIdToVisible[otherClient.csgoId];
            if (otherDistance < distance || (otherVisible && !targetVisible)) {
                targetVisible = otherVisible;
                nearestEnemyServerId = otherClient.csgoId;
                distance = otherDistance;
            }
        }
    }
    
    // if found any targets, set them, otherwise mark target as invalid
    if (nearestEnemyServerId != INVALID_ID) {
        int32_t targetCSKnowId = state.csgoIdToCSKnowId[nearestEnemyServerId];
        if (targetCSKnowId == executingPlan.target.csknowId && targetVisible) {
            developingPlan.numTimesRetargeted = executingPlan.numTimesRetargeted + 1;
        }
        else {
            developingPlan.numTimesRetargeted = 0;
        }
        developingPlan.target = {targetCSKnowId, {
            aimDis(aimGen) / std::pow(2, developingPlan.numTimesRetargeted),
            aimDis(aimGen) / std::pow(2, developingPlan.numTimesRetargeted)
        }, targetVisible};
    }
    else {
        developingPlan.target.csknowId = INVALID_ID;
    }
}

void Thinker::updateDevelopingPlanWaypoints(const Vec3 & curPosition, const Vec3 & targetPosition) {
    nav_mesh::vec3_t curPoint = vec3Conv(curPosition), 
        targetPoint = vec3Conv(targetPosition);

    auto optionalWaypoints = navFile.find_path(curPoint, targetPoint);
    if (optionalWaypoints) {
        developingPlan.waypoints = optionalWaypoints.value();
        if (developingPlan.waypoints.back() != targetPoint) {
            developingPlan.waypoints.push_back(targetPoint);
        }
    }
    // if waypoint finding fails, just walk randomnly
    else {
        navFile.find_path(curPoint, targetPoint);
        developingPlan.movementType = MovementType::Random;
    }
    developingPlan.curWaypoint = 0;
}

void Thinker::updateMovementType(const ServerState state, const ServerState::Client & curClient,
        const ServerState::Client & oldClient, const ServerState::Client & targetClient) {
    Vec3 curPosition{curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ};
    Vec3 oldPosition{oldClient.lastEyePosX, oldClient.lastEyePosY, oldClient.lastFootPosZ};

    Vec3 targetPosition;
    if (targetClient.csgoId != INVALID_ID) {
        targetPosition = {targetClient.lastEyePosX, targetClient.lastEyePosY, targetClient.lastFootPosZ};
    }
    else if (state.c4Exists) {
        targetPosition = {state.c4X, state.c4Y, state.c4Z}; 
    }
    else {
        // go to where you are if no where to go
        targetPosition = curPosition;
    }    

    // set random directions here so that random movement type chosen either due to 
    // nav path failure or getting stuck has new random directions to go in
    developingPlan.randomLeft = movementDis(movementGen) > 0.5;
    developingPlan.randomRight = !developingPlan.randomLeft;
    developingPlan.randomForward = movementDis(movementGen) > 0.5;
    developingPlan.randomBack = !developingPlan.randomForward;

    developingPlan.saveWaypoint = false;

    int numVisibleEnemies = 0, numVisibleTeammates = 0;
    std::map<int32_t, bool> csgoIdToVisible = getCSGOIdToVisibleClients(state, curClient);
    for (const ServerState::Client & otherClient : state.clients) {
        if (csgoIdToVisible[otherClient.csgoId] && otherClient.isAlive && curClient.csgoId != otherClient.csgoId) {
            if (otherClient.team == curClient.team) {
                numVisibleTeammates++;
            }
            else if (otherClient.team == 2 || otherClient.team == 3) {
                numVisibleEnemies++;
            }
        }
    }
    bool outnumbered = numVisibleEnemies > numVisibleTeammates + 1;

    if (skill.movementPolicy == MovementPolicy::PushOnly || 
            (skill.movementPolicy == MovementPolicy::PushAndRetreat && !outnumbered) ||
            (skill.movementPolicy == MovementPolicy::Normal && movementDis(movementGen) < 0.8)) {
        // set push, let updateDevelopingPlanWaypoint set to random if it fails
        developingPlan.movementType = MovementType::Push;
        // choose a new path only if no waypoints or target has changed or plan was for old round
        if (executingPlan.waypoints.empty() || executingPlan.waypoints.back() != vec3Conv(targetPosition) || 
                executingPlan.stateHistory.fromFront().roundNumber != developingPlan.stateHistory.fromFront().roundNumber) {
            updateDevelopingPlanWaypoints(curPosition, targetPosition);
        }
        // otherwise, keep same path
        else {
            developingPlan.waypoints = executingPlan.waypoints;
            developingPlan.saveWaypoint = true;
        }
    }
    else if (outnumbered) {
        // set retreat, let updateDevelopingPlanWaypoint set to random if it fails
        developingPlan.movementType = MovementType::Retreat;
        // if newly retreating, take oldest remembered position
        if (executingPlan.movementType != MovementType::Retreat) {
            // if get into a cycle of retreats that syncrhonizes with history length
            // taking oldest position will just sit still
            // this takes farthest
            size_t curRetreatOption = 0;
            double curRetreatDistance = computeDistance(curPosition, retreatOptions.fromFront());
            for (size_t i = 1; i < retreatOptions.getCurSize(); i++) {
                double nextRetreatDistance = computeDistance(curPosition, retreatOptions.fromFront(i));
                if (nextRetreatDistance > curRetreatDistance) {
                    curRetreatOption = i;
                    curRetreatDistance = nextRetreatDistance;
                }
            }
            /*
            if (retreatOptions.getCurSize() > 1 && curRetreatDistance < 20.) {
                std::raise(SIGINT);
            }
            */
            updateDevelopingPlanWaypoints(curPosition, retreatOptions.fromFront(curRetreatOption));
        }
        // if still retreating, make sure to get latest waypoint
        else {
            developingPlan.waypoints = executingPlan.waypoints;
            developingPlan.saveWaypoint = true;
        }
    }
    // walk randomly if stuck in a push or retreat but havent moved during window leading to plan decision
    else if ((executingPlan.movementType == MovementType::Push || executingPlan.movementType == MovementType::Retreat) &&
            computeDistance(curPosition, oldPosition) < 5.) {
        developingPlan.movementType = MovementType::Random;
    }
    else {
        developingPlan.movementType = MovementType::Hold;
    }

    /*
    if (developingPlan.movementType == MovementType::Random) {
        std::raise(SIGINT);
    }
    */

    // log results
    std::stringstream logStream;
    string targetName = executingPlan.target.csknowId != INVALID_ID ? 
        state.clients[executingPlan.target.csknowId].name : "";
    logStream << "num waypoints: " << developingPlan.waypoints.size() << ", cur movement type " 
        << enumAsInt(developingPlan.movementType) << ", num visible enemies " << numVisibleEnemies
        << ", target " << targetName << "\n";
    developingPlan.numLogLines++;

    if (!developingPlan.waypoints.empty()) {
        logStream << "first waypoint " << 0 << ":" 
            << developingPlan.waypoints[0].x << "," << developingPlan.waypoints[0].y 
            << "," << developingPlan.waypoints[0].z << "\n";

        uint64_t lastWaypoint = developingPlan.waypoints.size() - 1;
        logStream << "last waypoint " << lastWaypoint << ":" 
            << developingPlan.waypoints[lastWaypoint].x << "," << developingPlan.waypoints[lastWaypoint].y 
            << "," << developingPlan.waypoints[lastWaypoint].z << "\n";
        developingPlan.numLogLines += 2;
    }
    developingPlan.log += logStream.str();
}
