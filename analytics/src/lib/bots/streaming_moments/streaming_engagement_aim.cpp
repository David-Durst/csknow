//
// Created by durst on 12/25/22.
//

#include "bots/streaming_moments/streaming_engagement_aim.h"
#include "bots/analysis/vis_geometry.h"

namespace csknow::engagement_aim {
    EngagementAimTickData
    StreamingEngagementAim::computeOneTickData(StreamingBotDatabase & db,
                                               const fire_history::StreamingFireHistory & streamingFireHistory,
                                               CSGOId attackerId, const EngagementAimTarget & target,
                                               size_t attackerStateOffset, size_t victimStateOffset,
                                               bool firstEngagementTick, const VisPoints & visPoints) {

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
        Vec3 victimHeadPos =
            getCenterHeadCoordinatesForPlayer(victimEyePos, victimViewAngle, victimDuckAmount);
        if (firstEngagementTick) {
            playerToVictimEngagementFirstHeadPos[attackerId] = victimHeadPos;
        }

        const ServerState::Client & attackerClient =
            db.batchData.fromNewest(static_cast<int64_t>(attackerStateOffset))
            .getClient(attackerId);

        EngagementAimTickData engagementAimTickData;

        Vec3 attackerEyePos = attackerClient.getEyePosForPlayer();

        Vec2 curViewAngle = attackerClient.getCurrentViewAngles();

        curViewAngle.normalize();
        Vec2 idealViewAngle = viewFromOriginToDest(attackerEyePos, victimHeadPos);

        engagementAimTickData.idealViewAngle = idealViewAngle;

        engagementAimTickData.deltaRelativeFirstHeadViewAngle =
            deltaViewFromOriginToDest(attackerEyePos,
                                      playerToVictimEngagementFirstHeadPos[attackerId], curViewAngle);

        const fire_history::FireClientData & attackerFireData =
            streamingFireHistory.fireClientHistory.clientHistory.at(attackerId).fromNewest(attackerStateOffset);
        if (target.isPlayer()) {
            engagementAimTickData.hitVictim = attackerFireData.hitEnemy &&
                                              attackerFireData.victims.find(target.csgoId) !=
                                              attackerFireData.victims.end();
        }
        else {
            engagementAimTickData.hitVictim = false;
        }

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
        bool curTickVictimVisible = victimInFOV && victimVisNoFOV;
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
        Vec2 victimMinViewAngleCur{std::numeric_limits<double>::max(),
                                   std::numeric_limits<double>::max()};
        Vec2 victimMaxViewAngleCur{-1*std::numeric_limits<double>::max(),
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
    }

    void StreamingEngagementAim::addTickData(StreamingBotDatabase & db,
                     const fire_history::StreamingFireHistory & streamingFireHistory,
                     const VisPoints & visPoints) {
        const ServerState & curState = db.batchData.fromNewest();
        set<CSGOId> activeClients;

        for (const auto & curTickClient : curState.clients) {
            activeClients.insert(curTickClient.csgoId);
            const fire_history::FireClientData & curFireClientData =
                streamingFireHistory.fireClientHistory.clientHistory.at(curTickClient.csgoId).fromNewest();
            EngagementAimTickData engagementAimTickData;

            // all ticks need to know how far past they can look, as if not enough history then replicate available
            // data
            size_t oldestAttackerStateOffset = db.clientHistoryLength(curTickClient.csgoId);
            // history is irrelevant if target is a fixed position rather than a player
            size_t oldestVictimStateOffset = 0;
            const EngagementAimTarget & target = currentClientTargetMap.at(curTickClient.csgoId);
            if (target.isPlayer()) {
                oldestVictimStateOffset = db.clientHistoryLength(target.csgoId);
            }

            // if engagement is new, fill in all past
            if (priorClientTargetMap.find(curTickClient.csgoId) == priorClientTargetMap.end() ||
                priorClientTargetMap.at(curTickClient.csgoId) != target) {
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
                    computeOneTickData(db, streamingFireHistory, curTickClient.csgoId, target,
                                       attackerStateOffset, victimStateOffset,
                                       i == PAST_AIM_TICKS - 1);
                }
            }
            // now that past is filled in, fill in most recent state
            // no need for state offset shenanigans, always have current state
            computeOneTickData(db, streamingFireHistory, curTickClient.csgoId, target,
                               0, 0, false);


            engagementAimPlayerHistory.clientHistory.at(curTickClient.csgoId).enqueue(engagementAimTickData);
        }

        engagementAimPlayerHistory.removeInactiveClients(activeClients);
        priorClientTargetMap = currentClientTargetMap;
    }
}