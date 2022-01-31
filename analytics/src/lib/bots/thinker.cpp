#include "bots/thinker.h"
#include <limits>

float velocityCurve(double totalDeltaAngle, double lastDeltaAngle) {
    double newDeltaAngle = std::max(-1 * MAX_ONE_DIRECTION_ANGLE_VEL,
            std::min(MAX_ONE_DIRECTION_ANGLE_VEL, totalDeltaAngle / 3));
    double newAccelAngle = newDeltaAngle - lastDeltaAngle;
    if (std::abs(newAccelAngle) > MAX_ONE_DIRECTION_ANGLE_ACCEL) {
        newDeltaAngle = lastDeltaAngle + 
            copysign(MAX_ONE_DIRECTION_ANGLE_ACCEL, newAccelAngle);
    }
    return newDeltaAngle / MAX_ONE_DIRECTION_ANGLE_VEL;
}

Vec2 Thinker::aimAt(int targetClientId) {
    const ServerState::Client & curClient = state.clients[state.serverClientIdToCSKnowId[curBot]],
        & targetClient = state.clients[targetClientId];

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
    resultDeltaAngles.x = velocityCurve(totalDeltaAngles.x, lastDeltaAngles.x);
    resultDeltaAngles.y = velocityCurve(totalDeltaAngles.y, lastDeltaAngles.y);

    lastDeltaAngles = resultDeltaAngles;

    return resultDeltaAngles;
}

void Thinker::think() {
    if (curBot >= state.clients.size()) {
        return;
    }
    int csknowId = state.serverClientIdToCSKnowId[curBot];
    ServerState::Client & curClient = state.clients[csknowId];
    int nearestEnemyServerId = -1;
    double distance = std::numeric_limits<double>::max();
    for (const auto & otherClient : state.clients) {
        if (otherClient.team != curClient.team) {
            double otherDistance = computeDistance(
                    {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastEyePosZ},
                    {otherClient.lastEyePosX, otherClient.lastEyePosY, otherClient.lastEyePosZ});
            if (otherDistance < distance) {
                nearestEnemyServerId = otherClient.serverId;
                distance = otherDistance;
            }
        }
    }

    Vec2 angleDelta = 
        this->aimAt(state.serverClientIdToCSKnowId[nearestEnemyServerId]);

    state.inputsValid[csknowId] = true;
    bool attackLastFrame = curClient.buttons & IN_ATTACK > 0;
    curClient.buttons = 0;
    //curClient.buttons |= IN_FORWARD;
    curClient.inputAngleDeltaPctX = angleDelta.x;
    curClient.inputAngleDeltaPctY = angleDelta.y;

    ServerState::Client & targetClient = state.clients[state.serverClientIdToCSKnowId[nearestEnemyServerId]];
    Ray eyeCoordinates = getEyeCoordinatesForPlayer(
            {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ},
            {curClient.lastEyeWithRecoilAngleX, curClient.lastEyeWithRecoilAngleY});
    AABB targetAABB = getAABBForPlayer(
            {targetClient.lastEyePosX, targetClient.lastEyePosY, targetClient.lastFootPosZ});
    double hitt0, hitt1;
    if (intersectP(targetAABB, eyeCoordinates, hitt0, hitt1) && !attackLastFrame) {
        curClient.buttons |= IN_ATTACK;
    }
    //state.clients[csknowId].inputAngleDeltaPctX = 0.02;
    //state.clients[csknowId].inputAngleDeltaPctY = 0.;
}
