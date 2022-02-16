#include "bots/thinker.h"
#include <limits>

void Thinker::think() {
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
        planThread = std::move(std::thread(&Thinker::plan, this));
        launchedPlanThread = true;
    }

    // keep doing same thing if miss due to lock contention, don't block rest of system
    bool gotLock = planLock.try_lock();
    if (gotLock) {
        liveState.inputsValid[curBotCSKnowId] = true;
        curClient.buttons = 0;
        curClient.inputAngleDeltaPctX = 0.;
        curClient.inputAngleDeltaPctY = 0.;
        thinkLog = "Bot " + curClient.name + " CSGO id: " + std::to_string(curClient.csgoId) + 
            "\n" + executingPlan.log;
        numThinkLines = 1 + executingPlan.numLogLines;
        
        // only move if there's a plan
        if (executingPlan.valid) {
            // ensure that plan state mismatch doesn't cause crash
            if (liveState.clients.size() <= executingPlan.target.csknowId) {
                executingPlan.valid = false;
            }
            else {
                // assuming clients aren't reshuffled, if so, will just be off for a second
                const ServerState::Client & targetClient = 
                    liveState.clients[executingPlan.target.csknowId];
                // from back to get most recent prior client
                ServerState::Client & priorClient = getCurClient(stateForNextPlan.stateHistory.fromBack());
                this->aimAt(curClient, targetClient, executingPlan.target.offset, priorClient);
                this->fire(curClient, targetClient, priorClient);
                this->move(curClient, priorClient);
                this->defuse(curClient, targetClient);
            }
        }

        // clear the state history on each new round
        // take oldest as want if any are wrong, not just most recent
        if (stateForNextPlan.stateHistory.fromFront().roundNumber != liveState.roundNumber) {
            stateForNextPlan.stateHistory.clear();
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
        const Vec2 & aimOffset, const ServerState::Client & priorClient) {
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
    Vec2 targetAngles = vectorAngles(targetVector) + aimOffset;
    targetAngles.makePitchNeg90To90();
    targetAngles.y = std::max(-1 * MAX_PITCH_MAGNITUDE, 
            std::min(MAX_PITCH_MAGNITUDE, targetAngles.y));

    // https://stackoverflow.com/a/7428771
    Vec2 deltaAngles = targetAngles - currentAngles;
    deltaAngles.makeYawNeg180To180();

    curClient.inputAngleDeltaPctX = computeAngleVelocity(deltaAngles.x, priorClient.inputAngleDeltaPctX);
    curClient.inputAngleDeltaPctY = computeAngleVelocity(deltaAngles.y, priorClient.inputAngleDeltaPctY);
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
    

    Vec3 priorPos{priorClient.lastEyePosX, priorClient.lastEyePosY, priorClient.lastFootPosZ}; 
    Vec3 curPos{curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ};

    Ray eyeCoordinates = getEyeCoordinatesForPlayerGivenEyeHeight(
            {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastEyePosZ},
            {curClient.lastEyeWithRecoilAngleX, curClient.lastEyeWithRecoilAngleY});
    AABB targetAABB = getAABBForPlayer(
            {targetClient.lastEyePosX, targetClient.lastEyePosY, targetClient.lastFootPosZ});
    double hitt0, hitt1;
    bool aimingAtEnemy = intersectP(targetAABB, eyeCoordinates, hitt0, hitt1);
    // if the retreat will end up near the current position, no option but fight
    bool stopToRetreat = executingPlan.movementType == MovementType::Retreat && 
        computeDistance(curPos, vec3tConv(executingPlan.waypoints.back())) > 50.;
    inSpray = haveAmmo && visible && !stopToRetreat;
    
    bool stoppedEnoughToShoot = !skill.stopToShoot ||
        computeDistance(priorPos, curPos) < MAX_VELOCITY_WHEN_STOPPED;

    this->setButton(curClient, IN_ATTACK, 
            !attackLastFrame && aimingAtEnemy && inSpray && stoppedEnoughToShoot);
    this->setButton(curClient, IN_RELOAD, !haveAmmo);

    thinkLog += "eye coordinates: " + eyeCoordinates.toString() + "\n";
    thinkLog += "target AABB: " + targetAABB.toString() + "\n";
    thinkLog += "aiming at enemy: " + std::to_string(aimingAtEnemy ? 1 : 0) + "\n";
    thinkLog += "in spray: " + std::to_string(inSpray ? 1 : 0) + "\n";
    thinkLog += "stopped enough to shoot: " + std::to_string(stoppedEnoughToShoot ? 1 : 0) + "\n";
    numThinkLines += 5;
}

void Thinker::moveInDir(ServerState::Client & curClient, Vec2 dir) {
    // don't need to worry about targetAngles y since can't move up and down
    this->setButton(curClient, IN_FORWARD, 
            dir.x >= -80. && dir.x <= 80.);
    this->setButton(curClient, IN_MOVELEFT, 
            dir.x >= 10. && dir.x <= 170.);
    this->setButton(curClient, IN_BACK, 
            dir.x >= 100. || dir.x <= -100.);
    this->setButton(curClient, IN_MOVERIGHT, 
            dir.x >= -170. && dir.x <= -10.);
}

void Thinker::move(ServerState::Client & curClient, const ServerState::Client & priorClient) {
    Vec3 curPos{curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ};
    thinkLog += "cur pos: " + std::to_string(curPos.x) +
        "," + std::to_string(curPos.y) + "," + std::to_string(curPos.z) + "\n";
    numThinkLines++;

    Vec3 priorPos{priorClient.lastEyePosX, priorClient.lastEyePosY, priorClient.lastFootPosZ}; 

    if (executingPlan.movementType == MovementType::Hold) {
        this->setButton(curClient, IN_FORWARD, false);
        this->setButton(curClient, IN_MOVELEFT, false);
        this->setButton(curClient, IN_BACK, false);
        this->setButton(curClient, IN_MOVERIGHT, false);

        numThinkLines++;
        thinkLog += "Holding\n";
    }
    else if (executingPlan.movementType == MovementType::Random) {
        this->setButton(curClient, IN_FORWARD, executingPlan.randomForward);
        this->setButton(curClient, IN_MOVELEFT, executingPlan.randomLeft);
        this->setButton(curClient, IN_BACK, executingPlan.randomBack);
        this->setButton(curClient, IN_MOVERIGHT, executingPlan.randomRight);

        numThinkLines++;
        thinkLog += "Random movement\n";
    }
    else if (inSpray && skill.stopToShoot) {
        numThinkLines++;
        thinkLog += "Stopping to shoot\n";

        if (computeDistance(priorPos, curPos) < MAX_VELOCITY_WHEN_STOPPED) {
            // if not moving (roughly), then stop
            this->setButton(curClient, IN_FORWARD, false);
            this->setButton(curClient, IN_MOVELEFT, false);
            this->setButton(curClient, IN_BACK, false);
            this->setButton(curClient, IN_MOVERIGHT, false);
            return;
        }

        // when stopping to shoot, make sure to counter strafe
        Vec3 oppositeLastFrameVelocity = priorPos - curPos;        
        Vec2 targetAngles = vectorAngles(oppositeLastFrameVelocity);

        Vec2 currentAngles = {
            curClient.lastEyeAngleX + curClient.lastAimpunchAngleX, 
            curClient.lastEyeAngleY + curClient.lastAimpunchAngleY};

        Vec2 deltaAngles = targetAngles - currentAngles;
        deltaAngles.makeYawNeg180To180();
        moveInDir(curClient, deltaAngles);
    }
    else {
        Vec3 waypointPos{executingPlan.waypoints[executingPlan.curWaypoint].x,
            executingPlan.waypoints[executingPlan.curWaypoint].y,
            executingPlan.waypoints[executingPlan.curWaypoint].z};
        Vec3 targetVector = waypointPos - curPos;
        
        // crouch when shooting and moving
        if (inSpray) {
            this->setButton(curClient, IN_DUCK, true);
        }

       // if (computeMinDistanceLinePoint(priorPos, curPos, waypointPos) < 20.) {
        if (computeDistance(curPos, waypointPos) < 20.) {
            // move to next waypoint if not done path, otherwise stop
            if (executingPlan.curWaypoint < executingPlan.waypoints.size() - 1) {
                executingPlan.curWaypoint++;
            }
            else {
                numThinkLines++;
                thinkLog += "at path end\n";
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

        Vec2 deltaAngles = targetAngles - currentAngles;
        deltaAngles.makeYawNeg180To180();
        moveInDir(curClient, deltaAngles);

        if (!inSpray && std::abs(waypointPos.z - curPos.z) > 20.) {
            this->setButton(curClient, IN_JUMP, true);
        }

        std::stringstream thinkStream;
        if (executingPlan.curWaypoint < executingPlan.waypoints.size()) {
            thinkStream << "cur waypoint " << executingPlan.curWaypoint << ":"
                      << executingPlan.waypoints[executingPlan.curWaypoint].x << ","
                      << executingPlan.waypoints[executingPlan.curWaypoint].y << ","
                      << executingPlan.waypoints[executingPlan.curWaypoint].z << "\n";
            numThinkLines++;
        }

        thinkStream << "cur angle: " 
            << currentAngles.x << "," << currentAngles.y << "\n";
        numThinkLines++;

        thinkStream << "target angle: " 
            << targetAngles.x << "," << targetAngles.y << "\n";
        numThinkLines++;
        
        thinkLog += thinkStream.str();
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
