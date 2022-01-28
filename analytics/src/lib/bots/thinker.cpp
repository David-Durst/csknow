#include "bots/thinker.h"
#include <limits>

Vec2 Thinker::aimAt(int targetClientId) {
    const ServerState::Client & curClient = state.clients[state.serverClientIdToCSKnowId[curBot]],
        & targetClient = state.clients[targetClientId];

    Vec3 targetVector{
        targetClient.lastEyePosX - curClient.lastEyePosX,
        targetClient.lastEyePosY - curClient.lastEyePosY,
        targetClient.lastEyePosZ - curClient.lastEyePosZ
    };

    Vec2 currentAngles = {curClient.lastEyeAngleX, curClient.lastEyeAngleY};
    Vec2 targetAngles = vectorAngles(targetVector);
    targetAngles.makePitchNeg90To90();
    targetAngles.y = std::max(-1 * MAX_PITCH_MAGNITUDE, 
            std::min(MAX_PITCH_MAGNITUDE, targetAngles.y));

    Vec2 currentToTarget = targetAngles - currentAngles;
    Vec2 targetToCurrent = currentAngles - targetAngles;
    Vec2 deltaAngles = {
        std::abs(currentToTarget.x) <= std::abs(targetToCurrent.x) ?
            currentToTarget.x : targetToCurrent.x,
        std::abs(currentToTarget.y) <= std::abs(targetToCurrent.y) ?
            currentToTarget.y : targetToCurrent.y
    };
    deltaAngles.x /= MAX_ONE_DIRECTION_ANGLE_DELTA;
    deltaAngles.y /= MAX_ONE_DIRECTION_ANGLE_DELTA;
    deltaAngles = max({-1., -1}, min({1., 1.}, deltaAngles));

    return deltaAngles;
}

void Thinker::think() {
    if (curBot >= state.clients.size()) {
        return;
    }
    int csknowId = state.serverClientIdToCSKnowId[curBot];
    const ServerState::Client & curClient = state.clients[csknowId];
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
    state.clients[csknowId].buttons = 0;
    //state.clients[csknowId].buttons |= IN_FORWARD;
    state.clients[csknowId].inputAngleDeltaPctX = angleDelta.x;
    state.clients[csknowId].inputAngleDeltaPctY = angleDelta.y;
}
