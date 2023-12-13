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
                static_cast<double>(modelNavData.deltaXVal)/* * csknow::feature_store::delta_pos_grid_cell_dim)*/,
                static_cast<double>(modelNavData.deltaYVal)/* * csknow::feature_store::delta_pos_grid_cell_dim)*/,
                modelNavData.deltaZVal > 1 ? MAX_JUMP_HEIGHT : 0.
        };

        curPriority.targetAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();
        if (blackboard.removedAreas.find(curPriority.targetAreaId) != blackboard.removedAreas.end()) {
            curPriority.targetAreaId = blackboard.removedAreaAlternatives[curPriority.targetAreaId];
        }

        modelNavData.nextArea = curPriority.targetAreaId;
    }

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
        Vec3 scaledVel = movementStatus.getScaledVel(1.
                //static_cast<float>(inference_manager::ticks_per_inference) /
                //static_cast<float>(inference_manager::ticks_per_seconds)
        );
        modelNavData.deltaXVal = scaledVel.x;
        modelNavData.deltaYVal = scaledVel.y;
        modelNavData.deltaZVal = movementStatus.zBin;
        curPriority.moveOptions.walk = movementStatus.statureOption == weapon_speed::StatureOptions::Walking;
        curPriority.moveOptions.crouch = movementStatus.statureOption == weapon_speed::StatureOptions::Ducking;

        computeDeltaPosTargetPos(curClient, curPriority, modelNavData);
        modelNavData.unmodifiedTargetPos = curPriority.targetPos;
    }

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
                blackboard.playerToLastProbDeltaPosAssignment[treeThinker.csgoId] =
                        {Vec3{INVALID_ID, INVALID_ID, INVALID_ID}, 0, 0, false, false, false};
            }
            PriorityDeltaPosAssignment & lastProbDeltaPosAssignment =
                    blackboard.playerToLastProbDeltaPosAssignment[treeThinker.csgoId];
            if (!lastProbDeltaPosAssignment.valid) {
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
