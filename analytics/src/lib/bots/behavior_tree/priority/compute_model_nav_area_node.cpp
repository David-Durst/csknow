//
// Created by durst on 4/22/23.
//

#include "bots/analysis/learned_models.h"
#include "bots/behavior_tree/priority/compute_model_nav_area_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace csknow::compute_nav_area {

    void ComputeModelNavAreaNode::computeDeltaPosTargetPos(const ServerState::Client & curClient,
                                                           Priority &curPriority, ModelNavData &modelNavData) {
        // add half so get center of each area grid
        // no need for z since doing 2d compare
        curPriority.targetPos = curClient.getFootPosForPlayer() + Vec3{
                static_cast<double>(modelNavData.deltaXVal),
                static_cast<double>(modelNavData.deltaYVal),
                modelNavData.deltaZVal > 1 ? MAX_JUMP_HEIGHT : 0.
        };

        curPriority.targetAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();
        if (blackboard.removedAreas.find(curPriority.targetAreaId) != blackboard.removedAreas.end()) {
            curPriority.targetAreaId = blackboard.removedAreaAlternatives[curPriority.targetAreaId];
        }
        if (modelNavData.deltaZVal == 0) {
            blackboard.navFile.get_nearest_point_in_area(vec3Conv(curPriority.targetPos),
                                                         blackboard.navFile.get_area_by_id_fast(curPriority.targetAreaId));
        }

        modelNavData.nextArea = curPriority.targetAreaId;
    }

    // 11 blend points as first one will be starting point. want 10 points over entire length
    constexpr int num_blend_points = 11;

    void ComputeModelNavAreaNode::checkClearPathToTargetPos(const ServerState::Client & curClient,
                                                            Priority & curPriority, const ModelNavData &) {
        AreaId lastAreaId = INVALID_ID;
        // for each sample, verify that can walk from prior sample to next one
        // 1. base case - in the same nav mesh as in the prior one
        // 2. connected case - a nav mesh that shares an edge with the prior one
        //      a. since sampling at very fine level, this case shouldn't fail when 2 nav mesh cells are connected, but not at the location where a bot wants to cross (like rounding a corner)
        // 3. complex case - nav mesh that touches another navmesh
        //      c. this handles the case where navmesh cells form a 2x2 grid, and can't walk diagonally because those don't share an edge
        /*
        set<AreaId> badStartIds{4052};
        set<AreaId> badEndIds{7555};
        AreaId startAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curClient.getFootPosForPlayer())).get_id();
        AreaId endAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();
         */
        for (int blendIndex = 0; blendIndex < num_blend_points; blendIndex++) {
            float newFrac = static_cast<float>(blendIndex) / static_cast<float>(num_blend_points - 1);
            float oldFrac = 1.f - newFrac;
            Vec3 blendPoint = curClient.getFootPosForPlayer() * oldFrac + curPriority.targetPos * newFrac;

            // check if nav areas same
            // ok to over use strict nav mesh if falling off ledge, won't figure out if trace down or nearest for
            // each blend point
            //size_t navAboveBelowIndex = blackboard.navAboveBelow.posToIndex(blendPoint);
            AreaId curAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(blendPoint)).get_id();
            if (blendIndex == 0) {
                lastAreaId = curAreaId;
            }
            bool sameNavArea = lastAreaId == curAreaId;

            // check if nav areas connected
            int64_t lastAreaIndex = blackboard.mapMeshResult.areaToInternalId[lastAreaId];
            int64_t curAreaIndex = blackboard.mapMeshResult.areaToInternalId[curAreaId];
            bool connectedNavAreas = false;
            //if (modelNavData.deltaZVal != 1) {
            for (const auto & conId : blackboard.mapMeshResult.connectionAreaIds[lastAreaIndex]) {
                if (conId == curAreaId) {
                    connectedNavAreas = true;
                    break;
                }
            }
            //}

            // check if nav areas touching
            AABB lastAreaRegion = blackboard.mapMeshResult.coordinate[lastAreaIndex];
            AABB curAreaRegion = blackboard.mapMeshResult.coordinate[curAreaIndex];
            // expand one area region a little for tolerance
            lastAreaRegion.min.z -= MAX_OBSTACLE_SIZE;
            lastAreaRegion.max.z += MAX_OBSTACLE_SIZE;
            bool touchingNavAreas = aabbOverlap(lastAreaRegion, curAreaRegion);

            if (!sameNavArea && !connectedNavAreas && !touchingNavAreas) {
                /*
                if (curClient.name == "Steel") {
                    std::cout << "huh" << std::endl;
                }
                std::cout << curClient.name << " not valid path" << std::endl;
                 */
                curPriority.directPathToLearnedTargetPos = false;
                if (false && curClient.name == "Kyle") {
                    vector<Vec3> blendPoints;
                    std::cout << blendIndex << " ; ";
                    for (int blendIndex = 0; blendIndex < num_blend_points; blendIndex++) {
                        float newFrac = static_cast<float>(blendIndex) / static_cast<float>(num_blend_points - 1);
                        float oldFrac = 1.f - newFrac;
                        Vec3 tmpBlendPoint = curClient.getFootPosForPlayer() * oldFrac + curPriority.targetPos * newFrac;
                        AreaId tmpAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(tmpBlendPoint)).get_id();
                        std::cout << tmpBlendPoint.toCSV() << " " << tmpAreaId << " ; ";
                    }
                    std::cout << lastAreaId << ";" << curAreaId << std::endl;
                }
                /*
                if (lastAreaId == 8107 || lastAreaId == 4057 || lastAreaId == 4118 || lastAreaId == 4119 || lastAreaId == 4052) {
                    std::cout << "weird" << std::endl;
                }
                if (badStartIds.count(startAreaId) && badEndIds.count(endAreaId)) {
                    std::cout << "more weird" << std::endl;
                }
                 */
                return;
            }
            /*
            if (lastAreaId == 8107 && (curAreaId == 4138 || curAreaId == 4137 || curAreaId == 1414 || curAreaId == 7581)) {
                std::cout << "bad" << std::endl;
            }
             */
            lastAreaId = curAreaId;
        }
        //set<AreaId> badStartIds{8107, 4118, 4119, 4057, 4052, 7557};
        //set<AreaId> badEndIds{7581, 4138, 4139, 1414};
        /*
        if (badStartIds.count(startAreaId) && badEndIds.count(endAreaId)) {
            std::cout << "another bad" << std::endl;
        }
         */
        curPriority.directPathToLearnedTargetPos = true;
    }

    int validDest = 0, invalidDest = 0;
    void ComputeModelNavAreaNode::computeDeltaPosProbabilistic(const ServerState & state, Priority & curPriority,
                                                               CSGOId csgoId, ModelNavData & modelNavData) {
        //TargetPlayer & curTarget = curPriority.targetPlayer;
        // compute area probabilities
        const csknow::inference_delta_pos::InferenceDeltaPosPlayerAtTickProbabilities & deltaPosProbabilities =
                //curTarget.playerId == INVALID_ID ?
                blackboard.inferenceManager.playerToInferenceData.at(csgoId).deltaPosProbabilities; // :
                //blackboard.inferenceManager.playerToInferenceData.at(csgoId).combatDeltaPosProbabilities;
        vector<float> probabilities = deltaPosProbabilities.radialVelProbabilities;
        const ServerState::Client & curClient = state.getClient(csgoId);

        if (true || !curPriority.directPathToLearnedTargetPos || curPriority.directPathToLearnedTargetPos.value() ||
            curPriority.numConsecutiveLearnedPathOverrides > 2) {
            curPriority.numConsecutiveLearnedPathOverrides = 0;
            size_t deltaPosOption = 0;
            bool setDeltaPosOption = false;
            double probSample = blackboard.aggressionDis(blackboard.gen);
            double weightSoFar = 0.;
            modelNavData.deltaPosProbs.clear();
            for (size_t i = 0; i < probabilities.size(); i++) {
                weightSoFar += probabilities[i];
                modelNavData.deltaPosProbs.push_back(probabilities[i]);
                if (probSample < weightSoFar && !setDeltaPosOption) {
                    deltaPosOption = i;
                    setDeltaPosOption = true;
                }
            }
            // * /
            /*
            double maxProb = -1.;
            modelNavData.deltaPosProbs.clear();
            for (size_t i = 0; i < probabilities.size(); i++) {
                modelNavData.deltaPosProbs.push_back(probabilities[i]);
                if (probabilities[i] > maxProb) {
                    deltaPosOption = i;
                    maxProb = probabilities[i];
                }
            }
            */
            modelNavData.radialVelIndex = deltaPosOption;
            csknow::weapon_speed::MovementStatus movementStatus(static_cast<EngineWeaponId>(curClient.currentWeaponId),
                                                                curClient.isScoped, modelNavData.radialVelIndex);
            Vec3 scaledVel = movementStatus.getScaledVel(//1.
                    static_cast<float>(inference_manager::ticks_per_inference) /
                    static_cast<float>(inference_manager::ticks_per_seconds)
            );
            modelNavData.deltaXVal = scaledVel.x;
            modelNavData.deltaYVal = scaledVel.y;
            modelNavData.deltaZVal = movementStatus.zBin;
            curPriority.moveOptions.walk = movementStatus.statureOption == weapon_speed::StatureOptions::Walking;
            curPriority.moveOptions.crouch = movementStatus.statureOption == weapon_speed::StatureOptions::Ducking;
            curPriority.learnedJump = movementStatus.jumping;
            curPriority.learnedStop = !movementStatus.moving;

            computeDeltaPosTargetPos(curClient, curPriority, modelNavData);
            validDest++;
        }
        else {
            curPriority.numConsecutiveLearnedPathOverrides++;
            invalidDest++;
        }
        checkClearPathToTargetPos(curClient, curPriority, modelNavData);
        modelNavData.unmodifiedTargetPos = curPriority.targetPos;
        //std::cout << validDest / (static_cast<float>(invalidDest + validDest)) << std::endl;
    }

    //std::chrono::time_point<std::chrono::system_clock> lastInferenceTime;
    NodeState ComputeModelNavAreaNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);

        bool ctAlive = false, tAlive = false;
        for (const auto & client : state.clients) {
            if (client.isAlive && client.team == ENGINE_TEAM_CT) {
                ctAlive = true;
            }
            if (client.isAlive && client.team == ENGINE_TEAM_T) {
                tAlive = true;
            }
        }
        bool bothTeamsAlive = ctAlive && tAlive;

        // default values are set to invalid where necessary, so this is fine
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        if (blackboard.inAnalysis || !getPlaceAreaModelProbabilities(curClient.team) ||
            !blackboard.inferenceManager.haveValidData() || !bothTeamsAlive) {
            curPriority.learnedTargetPos = false;
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }

        curPriority.learnedTargetPos = true;

        const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(
            vec3Conv(state.getClient(treeThinker.csgoId).getFootPosForPlayer()));
        AreaId curAreaId = curArea.get_id();
        const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);

        /*
        if (state.getClient(treeThinker.csgoId).team == ENGINE_TEAM_T) {
            std::cout << state.getClient(treeThinker.csgoId).name << " order waypoints: ";
            for (const auto & waypoint : curOrder.waypoints) {
                std::cout << waypoint.placeName << ", ";
            }
            std::cout << std::endl;
        }
         */

        // if still in engagement, then this isn't reason to switch
        bool wasInEngagement = false;
        if (!inEngagePath) {
            wasInEngagement = curPriority.priorityType == PriorityType::Engagement;
            curPriority.priorityType = PriorityType::Order;
            curPriority.targetPlayer.playerId = INVALID_ID;
            curPriority.nonDangerAimArea = {};
            curPriority.shootOptions = ShootOptions::DontShoot;
        }
        curPriority.moveOptions = {true, false, false};

        // if put in the model orders but not ready for this player, just stand still
        if (blackboard.inferenceManager.playerToInferenceData.find(treeThinker.csgoId) ==
            blackboard.inferenceManager.playerToInferenceData.end() ||
            !blackboard.inferenceManager.playerToInferenceData.at(treeThinker.csgoId).validData) {
            curPriority.targetPos = state.getClient(treeThinker.csgoId).getFootPosForPlayer();
            curPriority.targetAreaId = curAreaId;
        }
        else {
            ModelNavData & modelNavData = blackboard.playerToModelNavData[treeThinker.csgoId];

            modelNavData.deltaPosMode = true;
            // update area
            bool timeForNewDeltaPos = blackboard.inferenceManager.ranDeltaPosInferenceThisTick || wasInEngagement;
            if (blackboard.playerToLastProbDeltaPosAssignment.find(treeThinker.csgoId) ==
                blackboard.playerToLastProbDeltaPosAssignment.end() || timeForNewDeltaPos) {
                //std::cout << "invalidating player to last prob delta pos assignment" << std::endl;
                blackboard.playerToLastProbDeltaPosAssignment[treeThinker.csgoId] =
                        {Vec3{INVALID_ID, INVALID_ID, INVALID_ID}, 0, 0, false, false, false};
            }
            PriorityDeltaPosAssignment & lastProbDeltaPosAssignment =
                    blackboard.playerToLastProbDeltaPosAssignment[treeThinker.csgoId];
            if (!lastProbDeltaPosAssignment.valid) {
                //auto start = std::chrono::system_clock::now();
                //std::chrono::duration<double> inferenceTime = start - lastInferenceTime;
                //std::cout << "treeThinker " << treeThinker.csgoId << "times between inferences " << inferenceTime.count() << std::endl;
                //lastInferenceTime = start;
                computeDeltaPosProbabilistic(state, curPriority, treeThinker.csgoId, modelNavData);
                lastProbDeltaPosAssignment = {curPriority.targetPos, curPriority.targetAreaId, modelNavData.radialVelIndex,
                                              curPriority.moveOptions.walk, curPriority.moveOptions.crouch, true};
                lastProbDeltaPosAssignment.valid = true;
            }
            else {
                curPriority.targetPos = lastProbDeltaPosAssignment.targetPos;
                curPriority.targetAreaId = lastProbDeltaPosAssignment.targetAreaId;
                curPriority.moveOptions.walk = lastProbDeltaPosAssignment.walk;
                curPriority.moveOptions.crouch = lastProbDeltaPosAssignment.crouch;
            }

            curPriority.nonDangerAimAreaType = NonDangerAimAreaType::Path;
            if (curClient.team == ENGINE_TEAM_T) {
                if (curOrder.playerToHoldIndex.count(curClient.csgoId) > 0) {
                    AreaId chokeAreaId = curOrder.holdIndexToChokeAreaId.find(
                            curOrder.playerToHoldIndex.find(treeThinker.csgoId)->second)->second;
                    if (blackboard.visPoints.isVisibleAreaId(curAreaId, chokeAreaId)) {
                        curPriority.nonDangerAimArea = chokeAreaId;
                        curPriority.nonDangerAimAreaType = NonDangerAimAreaType::Hold;
                    }
                }
            }
            // if in the target area (and not moving to c4), don't move
            /*
            if ((!blackboard.isPlayerDefuser(treeThinker.csgoId) || state.c4IsDefused) && curAreaId == curPriority.targetAreaId) {
                curPriority.moveOptions = {false, false, false};
            }
             */
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
