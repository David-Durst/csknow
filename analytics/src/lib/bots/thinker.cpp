#include "bots/thinker.h"
#include <limits>

void Thinker::think() {
    liveState.numThinkLines = 0;
    int32_t curBotCSKnowId = liveState.csgoIdToCSKnowId[curBotCSGOId];

    if (curBotCSGOId >= liveState.csgoIdToCSKnowId.size() ||
        curBotCSKnowId == -1) {
        return;
    }
    ServerState::Client & curClient = getCurClient(liveState);

    // don't think if dead or not bot
    if (!curClient.isAlive || !curClient.isBot) {
        liveState.inputsValid[liveState.csgoIdToCSKnowId[curBotCSGOId]] = false;
        return;
    }

    if (!launchedPlanThread) {
        planThread = std::thread(&Thinker::plan, this);
    }

    // keep doing same thing if miss due to lock contention, don't block rest of system
    bool gotLock = planLock.try_lock();
    if (gotLock) {
        liveState.inputsValid[curBotCSKnowId] = true;
        curClient.buttons = 0;
        curClient.inputAngleDeltaPctX = 0.;
        curClient.inputAngleDeltaPctY = 0.;
        thinkLog = executingPlan.log;
        numThinkLines = executingPlan.numLogLines;

        // only move if there's a plan
        if (executingPlan.valid) {
            // assuming clients aren't reshuffled, if so, will just be off for a second
            const ServerState::Client & targetClient = 
                liveState.clients[executingPlan.target.csknowId];
            ServerState::Client & priorClient = getCurClient(stateForNextPlan.stateHistory.fromFront());
            this->aimAt(curClient, targetClient, executingPlan.target.offset, priorClient);
            this->fire(curClient, targetClient, priorClient);
            this->move(curClient);
            this->defuse(curClient, targetClient);
        }

        // other thinkers may update inputs later
        // but only know your inputs, so doesn't matter
        stateForNextPlan.stateHistory.enqueue(liveState, true);

        planLock.unlock();
    }

    // this ensures repeating to log if repeating operations due to lock miss
    liveState.thinkLog += thinkLog;
    liveState.numThinkLines += numThinkLines;
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


void Thinker::aimAt(ServerState::Client & curClient, const ServerState::Client & targetClient,
        const Vec3 & aimOffset, const ServerState::Client & priorClient) {
    // if no plant and no enemies, then nothing to look at
    if (targetClient.csgoId == INVALID_ID && !liveState.c4IsPlanted) {
        curClient.inputAngleDeltaPctX = 0.;
        curClient.inputAngleDeltaPctY = 0.;
        return;
    }

    // look at an enemy if it exists, otherwise look at bomb
    Vec3 targetVector;
    if (targetClient.csgoId != INVALID_ID) {
        targetVector = {
            targetClient.lastEyePosX - curClient.lastEyePosX,
            targetClient.lastEyePosY - curClient.lastEyePosY,
            targetClient.lastEyePosZ - curClient.lastEyePosZ
        };
        targetVector = targetVector + aimOffset;
    }
    else {
        targetVector = { 
            liveState.c4X - curClient.lastEyePosX,
            liveState.c4Y - curClient.lastEyePosY,
            liveState.c4Z - curClient.lastEyePosZ
        };
    }

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
    resultDeltaAngles.x = computeAngleVelocity(totalDeltaAngles.x, priorClient.inputAngleDeltaPctX);
    resultDeltaAngles.y = computeAngleVelocity(totalDeltaAngles.y, priorClient.inputAngleDeltaPctY);

    curClient.inputAngleDeltaPctX = resultDeltaAngles.x;
    curClient.inputAngleDeltaPctY = resultDeltaAngles.y;
}

void Thinker::fire(ServerState::Client & curClient, const ServerState::Client & targetClient,
                   const ServerState::Client & priorClient) {
        // don't shoot or reload if no enemies left
    if (targetClient.csgoId == INVALID_ID) {
        this->setButton(curClient, IN_ATTACK, false);
        this->setButton(curClient, IN_RELOAD, false);
        inSpray = false;
        return;
    }

    bool attackLastFrame = priorClient.buttons & IN_ATTACK > 0;

    // no reloading for knives and grenades
    bool haveAmmo = true;
    if (curClient.currentWeaponId == curClient.rifleId) {
        haveAmmo = curClient.rifleClipAmmo > 0;
    }
    else if (curClient.currentWeaponId == curClient.pistolId) {
        haveAmmo = curClient.pistolClipAmmo > 0;
    }

    bool visible = liveState.visibilityClientPairs.find({
            std::min(curClient.csgoId, targetClient.csgoId), 
            std::max(curClient.csgoId, targetClient.csgoId)
        }) != liveState.visibilityClientPairs.end();
    

    Ray eyeCoordinates = getEyeCoordinatesForPlayer(
            {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ},
            {curClient.lastEyeWithRecoilAngleX, curClient.lastEyeWithRecoilAngleY});
    AABB targetAABB = getAABBForPlayer(
            {targetClient.lastEyePosX, targetClient.lastEyePosY, targetClient.lastFootPosZ});
    double hitt0, hitt1;
    inSpray = intersectP(targetAABB, eyeCoordinates, hitt0, hitt1) && haveAmmo && visible;
    this->setButton(curClient, IN_ATTACK, 
            !attackLastFrame && inSpray);
    this->setButton(curClient, IN_RELOAD, !haveAmmo);
}

void Thinker::move(ServerState::Client & curClient) {
    if (executingPlan.movementType == MovementType::Hold || inSpray) {
        this->setButton(curClient, IN_FORWARD, false);
        this->setButton(curClient, IN_MOVELEFT, false);
        this->setButton(curClient, IN_BACK, false);
        this->setButton(curClient, IN_MOVERIGHT, false);
    }
    else if (executingPlan.movementType == MovementType::Random) {
        this->setButton(curClient, IN_FORWARD, executingPlan.randomForward);
        this->setButton(curClient, IN_MOVELEFT, executingPlan.randomLeft);
        this->setButton(curClient, IN_BACK, executingPlan.randomBack);
        this->setButton(curClient, IN_MOVERIGHT, executingPlan.randomRight);
    }
    else {
        Vec3 waypointPos{executingPlan.waypoints[executingPlan.curWaypoint].x,
            executingPlan.waypoints[executingPlan.curWaypoint].y,
            executingPlan.waypoints[executingPlan.curWaypoint].z};
        Vec3 curPos{curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ};
        Vec3 targetVector = waypointPos - curPos;

        if (computeDistance(curPos, waypointPos) < 20.) {
            // move to next waypoint if not done path, otherwise stop
            if (executingPlan.curWaypoint < executingPlan.waypoints.size() - 1) {
                executingPlan.curWaypoint++;
            }
            else {
                this->setButton(curClient, IN_FORWARD, false);
                this->setButton(curClient, IN_MOVELEFT, false);
                this->setButton(curClient, IN_BACK, false);
                this->setButton(curClient, IN_MOVERIGHT, false);
                return;
            }
        }

        Vec2 currentAngles = {
            curClient.lastEyeAngleX + curClient.lastAimpunchAngleX, 
            curClient.lastEyeAngleY + curClient.lastAimpunchAngleY};
        Vec2 targetAngles = vectorAngles(targetVector);

        std::stringstream thinkStream;
        if (executingPlan.curWaypoint < executingPlan.waypoints.size()) {
            thinkStream << "cur waypoint " << executingPlan.curWaypoint << ":"
                      << executingPlan.waypoints[executingPlan.curWaypoint].x << ","
                      << executingPlan.waypoints[executingPlan.curWaypoint].y << ","
                      << executingPlan.waypoints[executingPlan.curWaypoint].z << "\n";
            numThinkLines++;
        }

        thinkStream << "cur point: "
            << curPos.x << "," << curPos.y 
            << "," << curPos.z << "\n";
        numThinkLines++;

        thinkStream << "cur angle: " 
            << currentAngles.x << "," << currentAngles.y << "\n";
        numThinkLines++;

        thinkStream << "target angle: " 
            << targetAngles.x << "," << targetAngles.y << "\n";
        numThinkLines++;
        
        thinkLog += thinkStream.str();

        Vec2 totalDeltaAngles = targetAngles - currentAngles;
        totalDeltaAngles.makeYawNeg180To180();
        // don't need to worry about targetAngles y since can't move up and down
        //
        this->setButton(curClient, IN_FORWARD, 
                totalDeltaAngles.x >= -80. && totalDeltaAngles.x <= 80.);
        this->setButton(curClient, IN_MOVELEFT, 
                totalDeltaAngles.x >= 10. && totalDeltaAngles.x <= 170.);
        this->setButton(curClient, IN_BACK, 
                totalDeltaAngles.x >= 100. || totalDeltaAngles.x <= -100.);
        this->setButton(curClient, IN_MOVERIGHT, 
                totalDeltaAngles.x >= -170. && totalDeltaAngles.x <= -10.);

        if (std::abs(waypointPos.z - curPos.z) > 20.) {
            this->setButton(curClient, IN_JUMP, true);
        }
    }
}

void Thinker::defuse(ServerState::Client & curClient, const ServerState::Client & targetClient) {
    // if near C4 and no one alive and it's planted, start defusing
    Vec3 curPos{curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ};
    Vec3 c4Pos{liveState.c4X, liveState.c4Y, liveState.c4Z};
    this->setButton(curClient, IN_USE, 
            targetClient.csgoId == INVALID_ID && liveState.c4IsPlanted &&
            computeDistance(curPos, c4Pos) < 50.);
}
