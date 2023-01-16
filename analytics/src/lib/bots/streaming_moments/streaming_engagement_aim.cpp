//
// Created by durst on 12/25/22.
//

#include "bots/streaming_moments/streaming_engagement_aim.h"
#include "bots/analysis/vis_geometry.h"
#include "bots/analysis/pytorch_utils.h"
#include <torch/script.h>

namespace csknow::engagement_aim {
    AimWeaponType weaponIdToWeaponType(int32_t weaponId) {
        switch (weaponId) {
            case enumAsInt(AimWeaponId::Deagle):
                return AimWeaponType::Pistol;
            case enumAsInt(AimWeaponId::Dualies):
                return AimWeaponType::Pistol;
            case enumAsInt(AimWeaponId::FiveSeven):
                return AimWeaponType::Pistol;
            case enumAsInt(AimWeaponId::Glock):
                return AimWeaponType::Pistol;
            case enumAsInt(AimWeaponId::AK):
                return AimWeaponType::AR;
            case enumAsInt(AimWeaponId::AUG):
                return AimWeaponType::AR;
            case enumAsInt(AimWeaponId::AWP):
                return AimWeaponType::Sniper;
            case enumAsInt(AimWeaponId::FAMAS):
                return AimWeaponType::AR;
            case enumAsInt(AimWeaponId::G3):
                return AimWeaponType::AR;
            case enumAsInt(AimWeaponId::Galil):
                return AimWeaponType::AR;
            case enumAsInt(AimWeaponId::M249):
                return AimWeaponType::Heavy;
            case enumAsInt(AimWeaponId::M4A4):
                return AimWeaponType::AR;
            case enumAsInt(AimWeaponId::Mac10):
                return AimWeaponType::SMG;
            case enumAsInt(AimWeaponId::P90):
                return AimWeaponType::SMG;
            case enumAsInt(AimWeaponId::MP5):
                return AimWeaponType::SMG;
            case enumAsInt(AimWeaponId::UMP):
                return AimWeaponType::SMG;
            case enumAsInt(AimWeaponId::XM1014):
                return AimWeaponType::Heavy;
            case enumAsInt(AimWeaponId::Bizon):
                return AimWeaponType::SMG;
            case enumAsInt(AimWeaponId::MAG7):
                return AimWeaponType::Heavy;
            case enumAsInt(AimWeaponId::Negev):
                return AimWeaponType::Heavy;
            case enumAsInt(AimWeaponId::SawedOff):
                return AimWeaponType::Heavy;
            case enumAsInt(AimWeaponId::Tec9):
                return AimWeaponType::Pistol;
            case enumAsInt(AimWeaponId::Zeus):
                return AimWeaponType::Unknown;
            case enumAsInt(AimWeaponId::P2000):
                return AimWeaponType::Pistol;
            case enumAsInt(AimWeaponId::MP7):
                return AimWeaponType::SMG;
            case enumAsInt(AimWeaponId::MP9):
                return AimWeaponType::SMG;
            case enumAsInt(AimWeaponId::Nova):
                return AimWeaponType::Heavy;
            case enumAsInt(AimWeaponId::P250):
                return AimWeaponType::Pistol;
            case enumAsInt(AimWeaponId::Scar):
                return AimWeaponType::AR;
            case enumAsInt(AimWeaponId::SG553):
                return AimWeaponType::AR;
            case enumAsInt(AimWeaponId::SSG):
                return AimWeaponType::Sniper;
            case enumAsInt(AimWeaponId::M4A1S):
                return AimWeaponType::AR;
            case enumAsInt(AimWeaponId::USPS):
                return AimWeaponType::Pistol;
            case enumAsInt(AimWeaponId::CZ):
                return AimWeaponType::Pistol;
            case enumAsInt(AimWeaponId::R8):
                return AimWeaponType::Pistol;
            default:
                return AimWeaponType::Unknown;
        }
    }

    EngagementAimTickData
    StreamingEngagementAim::computeOneTickData(StreamingBotDatabase & db,
                                               const fire_history::StreamingFireHistory & streamingFireHistory,
                                               CSGOId attackerId, const EngagementAimTarget & target,
                                               size_t attackerStateOffset, size_t victimStateOffset,
                                               bool firstEngagementTick) {

        // create victim position data depending on victim is player or target
        Vec3 victimFootPos;
        Vec3 victimEyePos;
        Vec2 victimViewAngle;
        Vec3 victimVel;
        double victimDuckAmount;
        bool victimAlive;

        if (target.isPlayer()) {
            // need to check for last frame where victim is alive
            const ServerState::Client & newestVictimClient =
                db.batchData.fromNewest(static_cast<int64_t>(victimStateOffset))
                    .getClient(target.csgoId);
            // addTickData ensures that playerToVictimLastAlivePos has value, just check if that value is latest
            StreamingPinId lastAliveStateId = playerToVictimLastAlivePos.at(attackerId);
            if (newestVictimClient.isAlive) {
                if (newestVictimClient.lastFrame > db.pinnedData.at(lastAliveStateId).getLastFrame()) {
                    db.unpinState(lastAliveStateId);
                    playerToVictimLastAlivePos.insert_or_assign(attackerId,
                                                                db.pinState(victimStateOffset));
                }
                victimEyePos = newestVictimClient.getEyePosForPlayer();
                victimFootPos = newestVictimClient.getFootPosForPlayer();
                victimViewAngle = newestVictimClient.getCurrentViewAngles();
                victimVel = newestVictimClient.getVelocity();
                victimDuckAmount = static_cast<double>(newestVictimClient.duckAmount);
                victimAlive = true;
            }
            // take last alive if dead
            else {
                const ServerState::Client & lastAliveVictimClient =
                    db.pinnedData.at(lastAliveStateId).getClient(target.csgoId);
                victimEyePos = lastAliveVictimClient.getEyePosForPlayer();
                victimFootPos = lastAliveVictimClient.getFootPosForPlayer();
                victimViewAngle = lastAliveVictimClient.getCurrentViewAngles();
                victimVel = {0., 0., 0.};
                victimDuckAmount = static_cast<double>(lastAliveVictimClient.duckAmount);
                victimAlive = false;
            }
        }
        else {
            victimEyePos = target.pos;
            victimFootPos = getFootCoordinatesForPlayerGivenEyePos(victimEyePos);
            victimViewAngle = {0, 0};
            victimVel = {0., 0., 0.};
            victimDuckAmount = 0.;
            victimAlive = false;
        }
        // need to flip eye angles from server to match train data
        Vec3 victimHeadPos =
            getCenterHeadCoordinatesForPlayer(victimEyePos, victimViewAngle, victimDuckAmount);
        if (firstEngagementTick) {
            playerToVictimEngagementFirstHeadPos[attackerId] = victimHeadPos;
        }

        const ServerState & serverStateAttackerOffset =
            db.batchData.fromNewest(static_cast<int64_t>(attackerStateOffset));
        const ServerState::Client & attackerClient =
            db.batchData.fromNewest(static_cast<int64_t>(attackerStateOffset))
            .getClient(attackerId);

        EngagementAimTickData engagementAimTickData;

        Vec3 attackerEyePos = attackerClient.getEyePosForPlayer();

        Vec2 curViewAngle;
        // try to live 1 frame in the future by taking last prediction, but take reality from game engine if not
        // possible
        if (!attackerClient.inputAngleDefined || attackerClient.forceInput) {
        //if (true || attackerStateOffset != 0 || playerToNewAngle.find(attackerId) == playerToNewAngle.end()) {
            curViewAngle = attackerClient.getCurrentViewAngles();
        }
        else {
            curViewAngle = {attackerClient.inputAngleX, attackerClient.inputAngleY};//playerToNewAngle[attackerId];
        }

        curViewAngle.normalize();
        Vec2 idealViewAngle = viewFromOriginToDest(attackerEyePos, victimHeadPos);

        engagementAimTickData.idealViewAngle = idealViewAngle;
        engagementAimTickData.attackerViewAngle = curViewAngle;

        engagementAimTickData.deltaRelativeFirstHeadViewAngle =
            deltaViewFromOriginToDest(attackerEyePos,
                                      playerToVictimEngagementFirstHeadPos[attackerId], curViewAngle);
        engagementAimTickData.deltaRelativeCurHeadViewAngle =
            deltaViewFromOriginToDest(attackerEyePos, victimHeadPos, curViewAngle);


        const fire_history::FireClientData & attackerFireData =
            streamingFireHistory.fireClientHistory.clientHistory.at(attackerId)
            .fromNewest(static_cast<int64_t>(attackerStateOffset));
        if (target.isPlayer()) {
            engagementAimTickData.hitVictim = attackerFireData.hitEnemy &&
                                              attackerFireData.victims.find(target.csgoId) !=
                                              attackerFireData.victims.end();
        }
        else {
            engagementAimTickData.hitVictim = false;
        }
        /*
        if (fakeAttackDist(gen) < 0.05) {
            engagementAimTickData.hitVictim = true;
        }
         */

        engagementAimTickData.recoilIndex = attackerClient.recoilIndex;

        // mul recoil by -1 as flipping all angles internally
        Vec2 recoil {
            attackerClient.lastAimpunchAngleX,
            -1 * attackerClient.lastAimpunchAngleY,
        };

        engagementAimTickData.scaledRecoilAngle = recoil * WEAPON_RECOIL_SCALE;

        engagementAimTickData.holdingAttack = attackerFireData.ticksSinceLastHoldingAttack == 0;
        engagementAimTickData.ticksSinceLastFire =
            std::min(MAX_TICKS_SINCE_LAST_FIRE_ATTACK, attackerFireData.ticksSinceLastFire);
        engagementAimTickData.ticksSinceLastHoldingAttack =
            std::min(MAX_TICKS_SINCE_LAST_FIRE_ATTACK, attackerFireData.ticksSinceLastHoldingAttack);


        /*
        vector<CellIdAndDistance> attackerCellIdsByDistances = visPoints.getCellVisPointsByDistance(
            attackerEyePos);
        vector<CellIdAndDistance> victimCellIdsByDistances = visPoints.getCellVisPointsByDistance(
            victimEyePos);
        vector<CellVisPoint> victimTwoClosestCellVisPoints = {
            visPoints.getCellVisPoints()[victimCellIdsByDistances[0].cellId],
            visPoints.getCellVisPoints()[victimCellIdsByDistances[1].cellId]
        };
        bool victimInFOV = getCellsInFOV(victimTwoClosestCellVisPoints, attackerEyePos,
                                         curViewAngle);
        // vis from either of attackers two closest cell vis points
        bool victimVisNoFOV = false;
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 2; j++) {
                victimVisNoFOV |= visPoints.getCellVisPoints()[attackerCellIdsByDistances[i].cellId]
                    .visibleFromCurPoint[victimCellIdsByDistances[j].cellId];
            }
        }
         */ //victimInFOV && victimVisNoFOV;
        // only alive if real player (csgoId defined), so only do visibility check if valid target.csgoId
        bool curTickVictimVisible = victimAlive &&
            serverStateAttackerOffset.isVisible(attackerId, target.csgoId);
        engagementAimTickData.victimVisible = curTickVictimVisible;
        // remove victim visibility tracking if new engagmeent
        if (firstEngagementTick &&
            playerToVictimFirstVisibleFrame.find(attackerId) != playerToVictimFirstVisibleFrame.end()) {
            playerToVictimFirstVisibleFrame.erase(attackerId);
        }
        // since always have attacker client, just use that frame
        if (curTickVictimVisible &&
            (playerToVictimFirstVisibleFrame.find(attackerId) == playerToVictimFirstVisibleFrame.end() ||
            playerToVictimFirstVisibleFrame.at(attackerId) > attackerClient.lastFrame)) {
            playerToVictimFirstVisibleFrame[attackerId] = attackerClient.lastFrame;
        }
        engagementAimTickData.victimVisibleYet =
            playerToVictimFirstVisibleFrame.find(attackerId) != playerToVictimFirstVisibleFrame.end() &&
            playerToVictimFirstVisibleFrame[attackerId] <= attackerClient.lastFrame;

        engagementAimTickData.victimAlive = victimAlive;

        AABB victimAABB = getAABBForPlayer(victimFootPos, victimDuckAmount);
        vector<Vec3> aabbCorners = getAABBCorners(victimAABB);
        Vec2 victimMinViewAngleFirstHead{std::numeric_limits<double>::max(),
                                         std::numeric_limits<double>::max()};
        Vec2 victimMaxViewAngleFirstHead{-1*std::numeric_limits<double>::max(),
                                         -1*std::numeric_limits<double>::max()};
        for (const auto & aabbCorner : aabbCorners) {
            Vec2 aabbViewAngle = viewFromOriginToDest(attackerEyePos, aabbCorner);
            Vec2 deltaAABBViewAngleFirstHead =
                deltaViewFromOriginToDest(attackerEyePos,
                                          playerToVictimEngagementFirstHeadPos[attackerId],
                                          aabbViewAngle);
            victimMinViewAngleFirstHead = min(victimMinViewAngleFirstHead, deltaAABBViewAngleFirstHead);
            victimMaxViewAngleFirstHead = max(victimMaxViewAngleFirstHead, deltaAABBViewAngleFirstHead);
        }

        engagementAimTickData.victimRelativeFirstHeadMinViewAngle = victimMinViewAngleFirstHead;
        engagementAimTickData.victimRelativeFirstHeadMaxViewAngle = victimMaxViewAngleFirstHead;
        engagementAimTickData.victimRelativeFirstHeadCurHeadViewAngle =
            deltaViewFromOriginToDest(attackerEyePos,
                                      playerToVictimEngagementFirstHeadPos[attackerId], idealViewAngle);

        engagementAimTickData.attackerEyePos = attackerEyePos;
        engagementAimTickData.victimEyePos = victimEyePos;
        engagementAimTickData.attackerVel = attackerClient.getVelocity();
        engagementAimTickData.victimVel = victimVel;
        return engagementAimTickData;
    }

    size_t numSkips = 0;
    void StreamingEngagementAim::predictNewAngles(const StreamingBotDatabase & db) {
        torch::NoGradGuard no_grad;
        const ServerState & curState = db.batchData.fromNewest();
        // record who doesn't have a target and which prediction index maps to which attacker id
        set<CSGOId> attackerIds;
        vector<CSGOId> orderedAttackerIds;
        std::vector<std::vector<float>> rowsCPP;
        auto options = torch::TensorOptions().dtype(at::kFloat);
        for (const auto & curTickClient : curState.clients) {
            if (currentClientTargetMap.find(curTickClient.csgoId) == currentClientTargetMap.end()) {
                continue;
            }
            attackerIds.insert(curTickClient.csgoId);
            orderedAttackerIds.push_back(curTickClient.csgoId);
            std::vector<float> rowCPP;
            // all but cur tick are inputs
            // seperate different input types
            const CircularBuffer<EngagementAimTickData> & engagementAimHistory =
                engagementAimPlayerHistory.clientHistory.at(curTickClient.csgoId);
            for (int64_t priorTickNum = PAST_AIM_TICKS - 1; priorTickNum >= 0; priorTickNum--) {
                const EngagementAimTickData & engagementAimTickData = engagementAimHistory.fromNewest(priorTickNum);
                /*
                rowCPP.push_back(static_cast<float>(boolToInt(engagementAimTickData.hitVictim)));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.recoilIndex));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.ticksSinceLastFire));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.ticksSinceLastHoldingAttack));
                 */
                rowCPP.push_back(static_cast<float>(boolToInt(engagementAimTickData.victimVisible)));
                //rowCPP.push_back(static_cast<float>(boolToInt(engagementAimTickData.victimVisibleYet)));
                rowCPP.push_back(static_cast<float>(boolToInt(engagementAimTickData.victimAlive)));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.attackerEyePos.x));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.attackerEyePos.y));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.attackerEyePos.z));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.victimEyePos.x));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.victimEyePos.y));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.victimEyePos.z));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.attackerVel.x));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.attackerVel.y));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.attackerVel.z));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.victimVel.x));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.victimVel.y));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.victimVel.z));
            }
            for (int64_t priorTickNum = PAST_AIM_TICKS - 1; priorTickNum >= 0; priorTickNum--) {
                const EngagementAimTickData & engagementAimTickData = engagementAimHistory.fromNewest(priorTickNum);
                rowCPP.push_back(static_cast<float>(engagementAimTickData.idealViewAngle.x));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.deltaRelativeFirstHeadViewAngle.x));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.scaledRecoilAngle.x));
                rowCPP.push_back(
                    static_cast<float>(engagementAimTickData.victimRelativeFirstHeadMinViewAngle.x));
                rowCPP.push_back(
                    static_cast<float>(engagementAimTickData.victimRelativeFirstHeadMaxViewAngle.x));
                rowCPP.push_back(
                    static_cast<float>(engagementAimTickData.victimRelativeFirstHeadCurHeadViewAngle.x));
            }
            for (int64_t priorTickNum = PAST_AIM_TICKS - 1; priorTickNum >= 0; priorTickNum--) {
                const EngagementAimTickData & engagementAimTickData = engagementAimHistory.fromNewest(priorTickNum);
                rowCPP.push_back(static_cast<float>(engagementAimTickData.idealViewAngle.y));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.deltaRelativeFirstHeadViewAngle.y));
                rowCPP.push_back(static_cast<float>(engagementAimTickData.scaledRecoilAngle.y));
                rowCPP.push_back(
                    static_cast<float>(engagementAimTickData.victimRelativeFirstHeadMinViewAngle.y));
                rowCPP.push_back(
                    static_cast<float>(engagementAimTickData.victimRelativeFirstHeadMaxViewAngle.y));
                rowCPP.push_back(
                    static_cast<float>(engagementAimTickData.victimRelativeFirstHeadCurHeadViewAngle.y));
            }
            for (int64_t priorTickNum = PAST_AIM_TICKS - 1; priorTickNum >= 0; priorTickNum--) {
                const EngagementAimTickData & engagementAimTickData = engagementAimHistory.fromNewest(priorTickNum);
                rowCPP.push_back(static_cast<float>(boolToInt(engagementAimTickData.holdingAttack)));
            }
            // TODO: handle weapons other than AK47
            rowCPP.push_back(
                static_cast<float>(enumAsInt(weaponIdToWeaponType(curTickClient.currentWeaponId))));
            rowsCPP.push_back(rowCPP);
        }
        if (!rowsCPP.empty()) {
            vector<torch::Tensor> rowsPT;
            for (auto & rowCPP : rowsCPP) {
                rowsPT.push_back(torch::from_blob(rowCPP.data(), {1, static_cast<long>(rowCPP.size())},
                                                  options));
            }
            torch::Tensor tmpTensor = torch::cat(rowsPT);
            std::vector<torch::jit::IValue> inputs{torch::cat(rowsPT)};
            const auto tuple_output = module.forward(inputs).toTuple();
            at::Tensor output = tuple_output->elements()[1].toTensor();
            at::Tensor raw_output = tuple_output->elements()[0].toTensor();

            for (size_t i = 0; i < orderedAttackerIds.size(); i++) {
                // subtract from input delta view angles to get change in angle, then apply that to current view angles
                //std::cout << output[0].size(0) << std::endl;
                Vec2 outputRelativeViewAngle = {
                    static_cast<double>(output[i][0].item<float>()),
                    static_cast<double>(output[i][output[0].size(0) / 3].item<float>())
                };
                playerToFiring[orderedAttackerIds[i]] =
                    static_cast<double>(output[i][output[0].size(0) * 2 / 3].item<float>());
                //const ServerState & prevState = db.batchData.fromNewest(1);
                EngagementAimTickData & newestTickData = engagementAimPlayerHistory.clientHistory.at(orderedAttackerIds[i])
                    .fromNewest();
                Vec2 inputRelativeViewAngle = newestTickData.deltaRelativeFirstHeadViewAngle;
                Vec2 deltaViewAngle = outputRelativeViewAngle - inputRelativeViewAngle;
                deltaViewAngle.y *= -1;
                deltaViewAngle.makePitchNeg90To90();
                deltaViewAngle.makeYawNeg180To180();
                Vec2 outputViewAngle = newestTickData.attackerViewAngle + deltaViewAngle;
                outputViewAngle.makeYawNeg180To180();
                EngagementAimTickData & recentTickData = engagementAimPlayerHistory.clientHistory.at(orderedAttackerIds[i])
                    .fromNewest(4);
                EngagementAimTickData & oldestTickData = engagementAimPlayerHistory.clientHistory.at(orderedAttackerIds[i])
                    .fromOldest();
                double distanceToTargetRecent =
                    computeMagnitude(newestTickData.deltaRelativeCurHeadViewAngle -
                                     recentTickData.deltaRelativeCurHeadViewAngle);
                double distanceToTargetOldest =
                    computeMagnitude(newestTickData.deltaRelativeCurHeadViewAngle -
                                     oldestTickData.deltaRelativeCurHeadViewAngle);
                if (distanceToTargetRecent < 0.75 && distanceToTargetOldest > 2. && newestTickData.recoilIndex < 2.) {
                    //playerToManualOverrideStart[orderedAttackerIds[i]] = db.batchData.fromNewest().getLastFrame();
                    playerToFiring[orderedAttackerIds[i]] = true;
                    deltaViewAngle = deltaViewAngle * 5;
                    outputViewAngle = newestTickData.attackerViewAngle + deltaViewAngle;
                    outputViewAngle.makeYawNeg180To180();
                }
                double mouseVelocityRecent =
                    computeMagnitude(newestTickData.attackerViewAngle - recentTickData.attackerViewAngle);
                double mouseVelocityOldest =
                    computeMagnitude(newestTickData.attackerViewAngle - recentTickData.attackerViewAngle);
                double enemyVelocityRecent = computeMagnitude(newestTickData.victimVel);
                if (mouseVelocityRecent < 0.75 && mouseVelocityOldest > 2. && enemyVelocityRecent > 50. && newestTickData.recoilIndex < 0.75) {
                    //playerToManualOverrideStart[orderedAttackerIds[i]] = db.batchData.fromNewest().getLastFrame();
                    playerToFiring[orderedAttackerIds[i]] = true;
                }
                /*
                if (newestTickData.recoilIndex > 7.) {
                    playerToFiring[orderedAttackerIds[i]] = false;
                }
                 */
                /*
                if (i == 0 && curState.getLastFrame() - prevState.getLastFrame() > 2) {
                    std::cout << "cur frame: " << curState.getLastFrame() << "prev frame: " << prevState.getLastFrame()
                        << "skips: " << numSkips
                        << "speed: " << computeMagnitude(deltaViewAngle)
                        << "cur view angle: " << newestTickData.attackerViewAngle.toString()
                        << "delta relative first head view angle: " << inputRelativeViewAngle.toString()
                        << "delta view angle: " << deltaViewAngle.toString()
                        << "cur pos: " << newestTickData.attackerEyePos.toString()
                        << "victim pos: " << newestTickData.victimEyePos.toString()
                        << std::endl;
                    numSkips++;
                }
                 */
                //EngagementAimTickData & priorTickData = engagementAimPlayerHistory.clientHistory.at(orderedAttackerIds[i])
                //    .fromNewest(1);
                //if (newestTickData.deltaRelativeFirstHeadViewAngle == priorTickData.deltaRelativeFirstHeadViewAngle) {
                bool loggedClient = curState.getClient(orderedAttackerIds[i]).team == ENGINE_TEAM_T &&
                    curState.getClient(orderedAttackerIds[i]).isAlive;
                if (loggedClient /*&& printAimTicks > 0*/) {
                    std::stringstream aimTicksStream;
                    aimTicksStream << curState.getClient(orderedAttackerIds[i]).name << std::endl;
                    aimTicksStream << curState.getClient(orderedAttackerIds[i]).getCurrentViewAngles().toString() << std::endl;
                    for (int64_t priorTickNum = PAST_AIM_TICKS - 1; priorTickNum >= 0; priorTickNum--) {
                        const auto s = engagementAimPlayerHistory.clientHistory.at(orderedAttackerIds[i])
                            .fromNewest(priorTickNum);
                        aimTicksStream << s.toCSV(priorTickNum == PAST_AIM_TICKS - 1) << std::endl;
                    }
                    /*
                    if (playerToFiring[orderedAttackerIds[i]]) {
                        std::cout << "firing" << std::endl;
                    }
                     */
                    torch::Tensor selectedTensor = tmpTensor.slice(0, i, i+1);
                    aimTicksStream << "input," << print2DTensor(selectedTensor);
                    torch::Tensor outputSelectedTensor = output.slice(0, i, i+1);
                    aimTicksStream << "output," << print2DTensor(outputSelectedTensor);
                    torch::Tensor outputRawSelectedTensor = raw_output.slice(0, i, i+1);
                    aimTicksStream << "output raw," << print2DTensor(outputRawSelectedTensor);
                    aimTicksStream << "distance to target recent," << distanceToTargetRecent << std::endl;
                    aimTicksStream << "distance to target oldest," << distanceToTargetOldest << std::endl;
                    //print2DTensor(tmpTensor);
                    aimTicksStream << "output view angle," << outputViewAngle.toString() << std::endl;
                    aimTicksStream << "delta view angle," << deltaViewAngle.toString() << std::endl;
                    aimTicksStream << "fire," << output[i][output[0].size(0) * 2 / 3].item<float>() << std::endl;
                    aimTicksStream << "fire fixed," << boolToString(playerToFiring[orderedAttackerIds[i]]) << std::endl;
                    aimTicksStream << "fire weights," <<
                        raw_output[i][raw_output[0].size(0) * 2 / 3].item<float>() << "," <<
                        raw_output[i][(raw_output[0].size(0) * 2 / 3) + 1].item<float>() << std::endl;
                    aimTicksStream << output[0].size(0) << std::endl;
                    aimTicksFile << aimTicksStream.str();
                }
                // flip y axis to go back to game coordinates
                playerToDeltaAngle[orderedAttackerIds[i]] = deltaViewAngle;
                playerToNewAngle[orderedAttackerIds[i]] = outputViewAngle;
            }
            // same angles as before if no target
            for (const auto & curTickClient : curState.clients) {
                if (attackerIds.find(curTickClient.csgoId) == attackerIds.end()) {
                    playerToDeltaAngle[curTickClient.csgoId] = {0., 0.};
                    playerToNewAngle[curTickClient.csgoId] = curTickClient.getCurrentViewAngles();
                    playerToFiring[curTickClient.csgoId] = false;
                }
            }
        }
        if (printAimTicks > 0) {
            printAimTicks--;
        }
    }

    void StreamingEngagementAim::addTickData(StreamingBotDatabase & db,
                     const fire_history::StreamingFireHistory & streamingFireHistory) {
        const ServerState & curState = db.batchData.fromNewest();
        set<CSGOId> activeClients;

        for (const auto & curTickClient : curState.clients) {
            activeClients.insert(curTickClient.csgoId);
            if (currentClientTargetMap.find(curTickClient.csgoId) == currentClientTargetMap.end()) {
                continue;
            }

            // all ticks need to know how far past they can look, as if not enough history then replicate available
            // data
            size_t oldestAttackerStateOffset = db.clientHistoryLength(curTickClient.csgoId) - 1;
            // history is irrelevant if target is a fixed position rather than a player
            size_t oldestVictimStateOffset = 0;
            const EngagementAimTarget & target = currentClientTargetMap.at(curTickClient.csgoId);
            if (target.isPlayer()) {
                oldestVictimStateOffset = db.clientHistoryLength(target.csgoId) - 1;
            }

            // if engagement is new or teleported (aka reset called), fill in all past
            if (priorClientTargetMap.find(curTickClient.csgoId) == priorClientTargetMap.end() ||
                priorClientTargetMap.at(curTickClient.csgoId) != target ||
                resetInternal) {
                /*
                if (curTickClient.lastTeleportId != curTickClient.lastTeleportConfirmationId) {
                    std::cout << curTickClient.name << " teleport id mismatch, id " <<
                    curTickClient.lastTeleportId << ", confirm id " << curTickClient.lastTeleportConfirmationId <<
                    ", view angle " << curTickClient.getCurrentViewAngles().toString() << std::endl;
                }
                 */
                engagementAimPlayerHistory.updateClient(curTickClient.csgoId);
                // removed pinned last victim alive from old victim
                if (playerToVictimLastAlivePos.find(curTickClient.csgoId) != playerToVictimLastAlivePos.end()) {
                    db.unpinState(playerToVictimLastAlivePos.at(curTickClient.csgoId));
                    playerToVictimLastAlivePos.erase(curTickClient.csgoId);
                }
                // add new victim, assuming victim alive on first tick
                playerToVictimLastAlivePos.insert({curTickClient.csgoId, db.pinState()});
                // do all but most recent state, as that will be filled in regardless if new or not
                for (size_t i = PAST_AIM_TICKS - 1; i > 0; i--) {
                    // try to get state i in past, but if don't have it, settle for oldest
                    size_t attackerStateOffset = std::min(i, oldestAttackerStateOffset);
                    size_t victimStateOffset = std::min(i, oldestVictimStateOffset);
                    engagementAimPlayerHistory.clientHistory.at(curTickClient.csgoId).enqueue(
                        computeOneTickData(db, streamingFireHistory, curTickClient.csgoId, target,
                                           attackerStateOffset, victimStateOffset,
                                           i == PAST_AIM_TICKS - 1));
                }
            }
            // now that past is filled in, fill in most recent state
            // no need for state offset shenanigans, always have current state
            engagementAimPlayerHistory.clientHistory.at(curTickClient.csgoId).enqueue(
                computeOneTickData(db, streamingFireHistory, curTickClient.csgoId, target,
                                   0, 0, false));
        }

        engagementAimPlayerHistory.removeInactiveClients(activeClients);
        predictNewAngles(db);
        /*
        for (const auto & curTickClient : curState.clients) {
            if (currentClientTargetMap.find(curTickClient.csgoId) == currentClientTargetMap.end()) {
                continue;
            }
            if (curTickClient.lastTeleportId != curTickClient.lastTeleportConfirmationId) {
                std::cout << "a" << std::endl;
                const auto &h = engagementAimPlayerHistory.clientHistory.at(curTickClient.csgoId);
                std::cout << "b" << std::endl;
                for (size_t i = 0; i < h.getCurSize(); i++) {
                    std::cout << i << " " << h.fromOldest(i).attackerViewAngle.toString() << std::endl;
                }
                std::cout << "c" << std::endl;
                std::cout << "new angle " << playerToNewAngle.at(curTickClient.csgoId).toString() << std::endl;
                std::cout << "delta angle " << playerToDeltaAngle.at(curTickClient.csgoId).toString() << std::endl;
            }
        }
         */
        priorClientTargetMap = currentClientTargetMap;
        resetInternal = false;
    }
}