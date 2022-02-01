#include "bots/thinker.h"
#include <limits>

void Thinker::think() {
    state.numThinkLines = 0;

    if (curBot >= state.clients.size()) {
        return;
    }
    int csknowId = state.serverClientIdToCSKnowId[curBot];
    ServerState::Client & curClient = state.clients[csknowId];

    // don't think if dead
    if (!curClient.isAlive) {
        return;
    }

    buttonsLastFrame = curClient.buttons;

    this->updatePolicy(curClient);

    Target target = selectTarget(curClient);
    const ServerState::Client & targetClient = state.clients[target.id];

    state.inputsValid[csknowId] = true;
    curClient.buttons = 0;

    this->aimAt(curClient, targetClient);
    this->fire(curClient, targetClient);
    this->move(curClient);
}

void Thinker::updatePolicy(const ServerState::Client & curClient) {
    auto curTime = std::chrono::system_clock::now();
    nav_mesh::vec3_t targetPoint{state.c4X, state.c4Y, state.c4Z}; 
    nav_mesh::vec3_t curPoint{curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ};
    std::chrono::duration<double> thinkTime = curTime - lastPolicyThinkTime;
    if (thinkTime.count() > SECONDS_BETWEEN_POLICY_CHANGES) {
        if (dis(gen) < 2.5 && state.c4IsPlanted) {
            curPolicy = PolicyStates::Push; 
            // choose a new path only if target has changed
            if (waypoints.empty() || waypoints.back() != targetPoint) {
                try {
                    waypoints = navFile.find_path(curPoint, targetPoint);                    
                }
                catch (std::exception& error) {
                    waypoints = {targetPoint};  
                }
                curWaypoint = 0;
            }
        }
        else {
            curPolicy = PolicyStates::Hold; 
        }
        lastPolicyThinkTime = curTime;
    }

    std::stringstream thinkStream;
    thinkStream << "num waypoints: " << waypoints.size() << ", cur policy " 
        << static_cast<std::underlying_type_t<PolicyStates>>(curPolicy) << "\n";
    state.numThinkLines++;

    if (waypoints.size() > curWaypoint) {
        thinkStream << "cur waypoint " << curWaypoint << ":" 
            << waypoints[curWaypoint].x << "," << waypoints[curWaypoint].y 
            << "," << waypoints[curWaypoint].z << "\n";
        state.numThinkLines++;
    }
    if (!waypoints.empty()) {
        uint64_t lastWaypoint = waypoints.size() - 1;
        thinkStream << "last waypoint " << lastWaypoint << ":" 
            << waypoints[lastWaypoint].x << "," << waypoints[lastWaypoint].y 
            << "," << waypoints[lastWaypoint].z << "\n";
        state.numThinkLines++;
    }
    state.thinkCopy = thinkStream.str();
}

Thinker::Target Thinker::selectTarget(const ServerState::Client & curClient) {
    int nearestEnemyServerId = -1;
    double distance = std::numeric_limits<double>::max();
    for (const auto & otherClient : state.clients) {
        if (otherClient.team != curClient.team && otherClient.isAlive) {
            double otherDistance = computeDistance(
                    {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastEyePosZ},
                    {otherClient.lastEyePosX, otherClient.lastEyePosY, otherClient.lastEyePosZ});
            if (otherDistance < distance) {
                nearestEnemyServerId = otherClient.serverId;
                distance = otherDistance;
            }
        }
    }
    
    return {state.serverClientIdToCSKnowId[nearestEnemyServerId], distance};
}

float computeAngleVelocity(double totalDeltaAngle, double lastDeltaAngle) {
    double newDeltaAngle = std::max(-1 * MAX_ONE_DIRECTION_ANGLE_VEL,
            std::min(MAX_ONE_DIRECTION_ANGLE_VEL, totalDeltaAngle / 3));
    double newAccelAngle = newDeltaAngle - lastDeltaAngle;
    if (std::abs(newAccelAngle) > MAX_ONE_DIRECTION_ANGLE_ACCEL) {
        newDeltaAngle = lastDeltaAngle + 
            copysign(MAX_ONE_DIRECTION_ANGLE_ACCEL, newAccelAngle);
    }
    return newDeltaAngle / MAX_ONE_DIRECTION_ANGLE_VEL;
}

void Thinker::aimAt(ServerState::Client & curClient, const ServerState::Client & targetClient) {
    Vec3 targetVector{
        targetClient.lastEyePosX - curClient.lastEyePosX,
        targetClient.lastEyePosY - curClient.lastEyePosY,
        targetClient.lastEyePosZ - curClient.lastEyePosZ
    };

    Vec2 currentAngles = {
        curClient.lastEyeAngleX + curClient.lastAimpunchAngleX, 
        curClient.lastEyeAngleY + curClient.lastAimpunchAngleY};
    Vec2 targetAngles = vectorAngles(targetVector);
    targetAngles.makePitchNeg90To90();
    targetAngles.y = std::max(-1 * MAX_PITCH_MAGNITUDE, 
            std::min(MAX_PITCH_MAGNITUDE, targetAngles.y));

    // https://stackoverflow.com/a/7428771
    Vec2 totalDeltaAngles = targetAngles - currentAngles;
    totalDeltaAngles.makeYawNeg180To180();

    Vec2 resultDeltaAngles;
    resultDeltaAngles.x = computeAngleVelocity(totalDeltaAngles.x, lastDeltaAngles.x);
    resultDeltaAngles.y = computeAngleVelocity(totalDeltaAngles.y, lastDeltaAngles.y);

    lastDeltaAngles = resultDeltaAngles;

    curClient.inputAngleDeltaPctX = resultDeltaAngles.x;
    curClient.inputAngleDeltaPctY = resultDeltaAngles.y;
}

void Thinker::fire(ServerState::Client & curClient, const ServerState::Client & targetClient) {
    bool attackLastFrame = buttonsLastFrame & IN_ATTACK > 0;

    // no reloading for knives and grenades
    bool haveAmmo = true;
    if (curClient.currentWeaponId == curClient.rifleId) {
        haveAmmo = curClient.rifleClipAmmo > 0;
    }
    else if (curClient.currentWeaponId == curClient.pistolId) {
        haveAmmo = curClient.pistolClipAmmo > 0;
    }
    //curClient.buttons |= IN_FORWARD;

    bool visible = state.visibilityClientPairs.find({ 
            std::min(curClient.serverId, targetClient.serverId), 
            std::max(curClient.serverId, targetClient.serverId)
        }) != state.visibilityClientPairs.end();
    

    Ray eyeCoordinates = getEyeCoordinatesForPlayer(
            {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ},
            {curClient.lastEyeWithRecoilAngleX, curClient.lastEyeWithRecoilAngleY});
    AABB targetAABB = getAABBForPlayer(
            {targetClient.lastEyePosX, targetClient.lastEyePosY, targetClient.lastFootPosZ});
    double hitt0, hitt1;
    this->setButton(curClient, IN_ATTACK, 
            intersectP(targetAABB, eyeCoordinates, hitt0, hitt1) && 
            !attackLastFrame && haveAmmo && visible);
    this->setButton(curClient, IN_RELOAD, !haveAmmo);
}

void Thinker::move(ServerState::Client & curClient) {
    //if (curPolicy == PolicyStates::Hold || getButton(curClient, IN_ATTACK)) {
    if (curPolicy == PolicyStates::Hold) {
        this->setButton(curClient, IN_FORWARD, false);
        this->setButton(curClient, IN_MOVERIGHT, false);
        this->setButton(curClient, IN_BACK, false);
        this->setButton(curClient, IN_MOVELEFT, false);
    }
    else {
        Vec3 waypointPos{waypoints[curWaypoint].x, waypoints[curWaypoint].y, waypoints[curWaypoint].z};
        Vec3 curPos{curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ};
        Vec3 targetVector = waypointPos - curPos;

        Vec2 currentAngles = {
            curClient.lastEyeAngleX + curClient.lastAimpunchAngleX, 
            curClient.lastEyeAngleY + curClient.lastAimpunchAngleY};
        Vec2 targetAngles = vectorAngles(targetVector);
        Vec2 totalDeltaAngles = targetAngles - currentAngles;
        totalDeltaAngles.makeYawNeg180To180();
        // don't need to worry about targetAngles y since can't move up and down
        //
        this->setButton(curClient, IN_FORWARD, 
                totalDeltaAngles.x >= 315. || totalDeltaAngles.x <= 45.);
        this->setButton(curClient, IN_MOVERIGHT, 
                totalDeltaAngles.x >= 45. && totalDeltaAngles.x <= 135.);
        this->setButton(curClient, IN_BACK, 
                totalDeltaAngles.x >= 135. && totalDeltaAngles.x <= 225.);
        this->setButton(curClient, IN_MOVELEFT, 
                totalDeltaAngles.x >= 225. && totalDeltaAngles.x <= 315.);

        if (computeDistance(curPos, waypointPos) < 20. && curWaypoint < waypoints.size() - 1) {
            curWaypoint++;
        }
    }
}
