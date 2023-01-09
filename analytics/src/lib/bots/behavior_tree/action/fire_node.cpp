//
// Created by durst on 5/8/22.
//
#include "bots/input_bits.h"
#include "bots/behavior_tree/action/action_node.h"
constexpr bool learned_firing = false;

namespace action {
    // const set<int32_t> scopedWeaponIds{AWP_ID, GSG_ID, SCAR_ID, SCOUT_ID};

    NodeState FireTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        Action & curAction = blackboard.playerToAction[treeThinker.csgoId];
        Action & oldAction = blackboard.lastPlayerToAction[treeThinker.csgoId];
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        //Path & curPath = blackboard.playerToPath[treeThinker.csgoId];

        //bool scopeResetTimePassed = state.getSecondsBetweenTimes(curAction.lastScopeTime, state.loadTime) > MIN_SCOPE_RESET_SECONDS;

        // don't shoot if there's no target
        if (curPriority.targetPlayer.playerId == INVALID_ID) {
            curAction.shotsInBurst = 0;
            curAction.setButton(IN_ATTACK, false);
            /*
            // un scope if not scoped in and haven't pressed scope button recently
            if (scopeResetTimePassed && curClient.isScoped) {
                curAction.lastScopeTime = state.loadTime;
                curAction.setButton(IN_ATTACK2, true);
            }
            */
        }
        else {
            /*
            // scope in if not scoped and haven't pressed scoped in recently
            if (scopeResetTimePassed && !curClient.isScoped &&
                scopedWeaponIds.find(curClient.currentWeaponId) != scopedWeaponIds.end()) {
                curAction.lastScopeTime = state.loadTime;
                curAction.setButton(IN_ATTACK2, true);
                std::cout << "scoping in" << std::endl;
            }
             */
            // shoot if the target is in our sights and recoil is sufficiently reset and have ammo and didn't fire
            // last frame (for pistols)

            // don't shoot if maxed out on shots in burst and there's recoil
            bool haveRecoil = (std::abs(curClient.lastEyeWithRecoilAngleX - curClient.lastEyeAngleX) > RECOIL_THRESHOLD) ||
                    (std::abs(curClient.lastEyeWithRecoilAngleY - curClient.lastEyeAngleY) > RECOIL_THRESHOLD);
            int maxBurstShots = 100;
            if (curPriority.shootOptions == ShootOptions::Tap) {
                maxBurstShots = 1;
            }
            else if (curPriority.shootOptions == ShootOptions::Burst) {
                maxBurstShots = 3;
            }

            if (curPriority.shootOptions == ShootOptions::DontShoot ||
                (haveRecoil && curAction.shotsInBurst > maxBurstShots)) {
                curAction.setButton(IN_ATTACK, false);
            }
            else {
                // reset burst if no recoil
                if (curAction.shotsInBurst > 0 && !haveRecoil) {
                    curAction.shotsInBurst = 0;
                }

                const ServerState::Client & targetClient = state.getClient(curPriority.targetPlayer.playerId);

                bool attackLastFrame = (oldAction.buttons & IN_ATTACK) > 0;

                bool haveAmmo = true;
                if (curClient.currentWeaponId == curClient.rifleId) {
                    haveAmmo = curClient.rifleClipAmmo > 0;
                }
                else if (curClient.currentWeaponId == curClient.pistolId) {
                    haveAmmo = curClient.pistolClipAmmo > 0;
                }

                // check if aiming at enemy anywhere
                Ray eyeCoordinates = getEyeCoordinatesForPlayerGivenEyeHeight(
                        curClient.getEyePosForPlayer(),
                        curClient.getCurrentViewAnglesWithAimpunch());

                AABB targetAABB = getAABBForPlayer(targetClient.getFootPosForPlayer());
                double hitt0, hitt1;
                bool aimingAtEnemy = intersectP(targetAABB, eyeCoordinates, hitt0, hitt1);
                if (learned_firing) {
                    const unordered_map<CSGOId, bool> & playerToFiring =
                        blackboard.streamingManager.streamingEngagementAim.playerToFiring;
                    if (playerToFiring.find(curClient.csgoId) == playerToFiring.end()) {
                        aimingAtEnemy = false;
                    }
                    else {
                        aimingAtEnemy = playerToFiring.at(curClient.csgoId);
                    }
                }

                // TODO: AFTER ADDING VELOCITY FIELD, TRACK STOPPED TO SHOOT USING VELOCITY
                curAction.setButton(IN_ATTACK, !attackLastFrame && haveAmmo && aimingAtEnemy);
                curAction.intendedToFire = aimingAtEnemy;
                if (curAction.getButton(IN_ATTACK)) {
                    /*
                    std::cout << "firing at " << targetClient.name << std::endl;
                    std::cout << "eye coordinates " << eyeCoordinates.toString() << std::endl;
                    std::cout << "targetAABB " << targetAABB.toString() << std::endl;
                    std::cout << "cur client name " << curClient.name << std::endl;
                    std::cout << "cur view angles " << curClient.getCurrentViewAngles().toString() << std::endl;
                    std::cout << "cur view angles with aimunch" << curClient.getCurrentViewAnglesWithAimpunch().toString() << std::endl;
                     */
                    curAction.shotsInBurst++;
                }
                /*
                if (!state.isVisible(curClient.csgoId, curPriority.targetPlayer.playerId) && curAction.getButton(IN_ATTACK)) {
                    int x = 1;
                }
                 */
                curAction.setButton(IN_RELOAD, !haveAmmo);
            }
        }
        if (blackboard.isPlayerDefuser(treeThinker.csgoId) && computeDistance(state.getC4Pos(), curClient.getFootPosForPlayer()) < DEFUSE_DISTANCE) {
            curAction.setButton(IN_USE, true);
        }

        playerNodeState[treeThinker.csgoId] = curAction.getButton(IN_ATTACK) ? NodeState::Success : NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }
}
