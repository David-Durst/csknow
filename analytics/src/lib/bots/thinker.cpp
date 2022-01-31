#include "bots/thinker.h"
#include <limits>

void Thinker::think() {
    if (curBot >= state.clients.size()) {
        return;
    }
    int csknowId = state.serverClientIdToCSKnowId[curBot];
    ServerState::Client & curClient = state.clients[csknowId];

    Target target = selectTarget(curClient);
    const ServerState::Client & targetClient = state.clients[target.id];

    state.inputsValid[csknowId] = true;

    this->aimAt(curClient, targetClient);
    this->fire(curClient, targetClient);

    //state.clients[csknowId].inputAngleDeltaPctX = 0.02;
    //state.clients[csknowId].inputAngleDeltaPctY = 0.;
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
    bool attackLastFrame = curClient.buttons & IN_ATTACK > 0;
    //curClient.buttons |= IN_FORWARD;

    Ray eyeCoordinates = getEyeCoordinatesForPlayer(
            {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ},
            {curClient.lastEyeWithRecoilAngleX, curClient.lastEyeWithRecoilAngleY});
    AABB targetAABB = getAABBForPlayer(
            {targetClient.lastEyePosX, targetClient.lastEyePosY, targetClient.lastFootPosZ});
    double hitt0, hitt1;
    if (intersectP(targetAABB, eyeCoordinates, hitt0, hitt1) && !attackLastFrame) {
        curClient.buttons |= IN_ATTACK;
    }
    else {
        curClient.buttons &= ~IN_ATTACK;
    }
}
