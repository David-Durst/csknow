//
// Created by durst on 5/8/22.
//
#include "bots/input_bits.h"
#include "bots/behavior_tree/action_node.h"

namespace action {
    NodeState FireTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        Action & curAction = blackboard.playerToAction[treeThinker.csgoId];
        Action & oldAction = blackboard.lastPlayerToAction[treeThinker.csgoId];
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        Path & curPath = blackboard.playerToPath[treeThinker.csgoId];

        // don't shoot if there's no target
        if (curPriority.targetPlayer.playerId == INVALID_ID) {
            curAction.shotsInBurst = 0;
            curAction.setButton(IN_ATTACK, false);
        }
        else {
            // shoot if the target is in our sights and recoil is sufficiently reset and have ammo and didn't fire
            // last frame (for pistols)

            // don't shoot if maxed out on shots in burst and there's recoil
            bool haveRecoil = (std::abs(curClient.lastEyeWithRecoilAngleX - curClient.lastEyeAngleX) > RECOIL_THRESHOLD) ||
                    (std::abs(curClient.lastEyeWithRecoilAngleY - curClient.lastEyeAngleY) > RECOIL_THRESHOLD);
            int maxBurstShots = 100;
            if (curPath.shootOptions == PathShootOptions::Tap) {
                maxBurstShots = 1;
            }
            else if (curPath.shootOptions == PathShootOptions::Burst) {
                maxBurstShots = 3;
            }

            if (haveRecoil && curAction.shotsInBurst > maxBurstShots) {
                curAction.setButton(IN_ATTACK, false);
            }
            else {
                // reset burst if no recoil
                if (!haveRecoil) {
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

                // TODO: AFTER ADDING VELOCITY FIELD, TRACK STOPPED TO SHOOT USING VELOCITY
                //curAction.setButton(IN_ATTACK, !attackLastFrame && haveAmmo && aimingAtEnemy);
                curAction.setButton(IN_RELOAD, !haveAmmo);
            }
        }

        playerNodeState[treeThinker.csgoId] = curAction.getButton(IN_ATTACK) ? NodeState::Success : NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }
}
