//
// Created by durst on 4/22/23.
//

#include "bots/analysis/learned_models.h"
#include "bots/behavior_tree/priority/compute_model_nav_area_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace csknow::compute_nav_area {
    constexpr double max_place_distance_seconds = 5.;

    Vec3 ComputeModelNavAreaNode::tryDeltaPosTargetPos(const ServerState & state, const ServerState::Client & curClient,
                                                       Priority &curPriority, ModelNavData &modelNavData) {
        // add half so get center of each area grid
        // no need for z since doing 2d compare
        curPriority.targetPos = curClient.getFootPosForPlayer() + Vec3{
                static_cast<double>(modelNavData.deltaXVal * csknow::feature_store::delta_pos_grid_cell_dim),
                static_cast<double>(modelNavData.deltaYVal * csknow::feature_store::delta_pos_grid_cell_dim),
                0.
        };


        /*
        curPriority.targetAreaId = blackboard.navFile
                .get_nearest_area_by_position_z_limit(vec3Conv(curPriority.targetPos), 2000.f, 2*MAX_JUMP_HEIGHT).get_id();
                */
        // check if closer jumping or not
        const nav_mesh::nav_area & flatArea = blackboard.navFile
                .get_nearest_area_by_position(vec3Conv(curPriority.targetPos));
        float flatDistance = blackboard.navFile.get_point_to_area_distance(vec3Conv(curPriority.targetPos), flatArea);
        Vec3 jumpingTargetPos = curPriority.targetPos;
        jumpingTargetPos.z += MAX_JUMP_HEIGHT;// * 0.85;
        const nav_mesh::nav_area & jumpingArea = blackboard.navFile
                .get_nearest_area_by_position(vec3Conv(jumpingTargetPos));
        float jumpingDistance = blackboard.navFile.get_point_to_area_distance(vec3Conv(jumpingTargetPos), jumpingArea);
        if (jumpingDistance < flatDistance) {
            curPriority.targetPos = jumpingTargetPos;
            curPriority.targetAreaId = jumpingArea.get_id();
        }
        else {
            curPriority.targetAreaId = flatArea.get_id();
        }

        Vec3 desiredPos = curPriority.targetPos;

        // if cur place is bombsite and CT defuser, then move to c4
        if (blackboard.isPlayerDefuser(curClient.csgoId) && !blackboard.inTest) {
            bool tAlive = false;
            for (const auto & client : state.clients) {
                if (client.isAlive && client.team == ENGINE_TEAM_T) {
                    tAlive = true;
                }
            }
            if (!tAlive) {
                curPriority.targetAreaId = blackboard.navFile
                        .get_nearest_area_by_position(vec3Conv(state.getC4Pos())).get_id();
            }
        }

        if (blackboard.removedAreas.find(curPriority.targetAreaId) != blackboard.removedAreas.end()) {
            curPriority.targetAreaId = blackboard.removedAreaAlternatives[curPriority.targetAreaId];
        }
        // if same next place and not at old next area, keep using that area
        /*
        if (blackboard.playerToModelNavData.find(csgoId) != blackboard.playerToModelNavData.end()) {
            const ModelNavData & oldModelNavData = blackboard.playerToModelNavData.at(csgoId);
            AreaId curAreaId = blackboard.navFile
                .get_nearest_area_by_position(vec3Conv(state.getClient(csgoId).getFootPosForPlayer())).get_id();
            if (oldModelNavData.nextPlace == modelNavData.nextPlace && oldModelNavData.nextArea != curAreaId) {
                curPriority.targetAreaId = oldModelNavData.nextArea;
            }
        }
        curPriority.targetPos = vec3tConv(blackboard.navFile.get_nearest_point_in_area(
                vec3Conv(curPriority.targetPos), blackboard.navFile.get_area_by_id_fast(curPriority.targetAreaId)));
         */
        modelNavData.nextArea = curPriority.targetAreaId;
        return desiredPos;
    }

    void ComputeModelNavAreaNode::computeDeltaPosProbabilistic(const ServerState & state, Priority & curPriority,
                                                               CSGOId csgoId, ModelNavData & modelNavData) {
        // compute area probabilities
        const csknow::inference_delta_pos::InferenceDeltaPosPlayerAtTickProbabilities & deltaPosProbabilities =
                blackboard.inferenceManager.playerToInferenceData.at(csgoId).deltaPosProbabilities;
        vector<float> probabilities = deltaPosProbabilities.deltaPosProbabilities;
        const ServerState::Client & curClient = state.getClient(csgoId);

        // re-weight just because want to be certain sums to one
        /*
        double reweightFactor = 0.;
        for (size_t i = 0; i < probabilities.size(); i++) {
            reweightFactor += probabilities[i];
        }
        for (size_t i = 0; i < probabilities.size(); i++) {
            probabilities[i] *= 1/reweightFactor;
        }
         */
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
        modelNavData.deltaPosIndex = deltaPosOption;

        // compute map grid to pos, and then pos to area
        // ok to pick bad area, as computePath in path node will pick a valid alternative (tree computes alternatives)
        modelNavData.deltaXVal = (deltaPosOption % csknow::feature_store::delta_pos_grid_num_cells_per_dim) -
                (csknow::feature_store::delta_pos_grid_num_cells_per_dim / 2);
        modelNavData.deltaYVal = (deltaPosOption / csknow::feature_store::delta_pos_grid_num_cells_per_dim) -
                (csknow::feature_store::delta_pos_grid_num_cells_per_dim / 2);

        if (modelNavData.deltaXVal == 0  && modelNavData.deltaYVal == 0) {
            int x = 1;
            (void) x;
        }

        AreaId priorTargetAreaId = curPriority.targetAreaId;
        Vec3 priorTargetPos = curPriority.targetPos;

        tryDeltaPosTargetPos(state, curClient, curPriority, modelNavData);

        /*
        // if same as prior target pos, try doubling distance and picking something else
        if (curPriority.targetPos.x == modelNavData.unmodifiedTargetPos.x &&
            curPriority.targetPos.y == modelNavData.unmodifiedTargetPos.y) {
            modelNavData.deltaXVal *= 2;
            modelNavData.deltaYVal *= 2;
            AreaId badAreaId = curPriority.targetAreaId;
            tryDeltaPosTargetPos(state, curClient, curPriority, modelNavData);
            if (badAreaId != curPriority.targetAreaId) {
                modelNavData.disabledArea = badAreaId;
            }
        }
        // if overrode target previously, don't go back, stick with fixed destination
        else if (curPriority.targetAreaId == modelNavData.disabledArea) {
            curPriority.targetPos = priorTargetPos;
            curPriority.targetAreaId = priorTargetAreaId;
            modelNavData.nextArea = curPriority.targetAreaId;
        }
        else {
            modelNavData.disabledArea = std::nullopt;
        }
         */

        /*
        if (priorTargetAreaId == 7566 && curPriority.targetAreaId == 7566) {
            std::cout << "repeat with prior target pos " << priorTargetPos.toString()
                << " to cur target pos " << curPriority.targetPos.toString()
                << " and prior unmodified target pos " << modelNavData.unmodifiedTargetPos.toString() << std::endl;
        }
         */
        if (priorTargetAreaId == 6803 && curPriority.targetAreaId == 8802) {
            std::cout << "jumping" << std::endl;
        }
        modelNavData.unmodifiedTargetPos = curPriority.targetPos;
    }

    NodeState ComputeModelNavAreaNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        if (blackboard.inAnalysis || !getPlaceAreaModelProbabilities(curClient.team) ||
            !blackboard.inferenceManager.haveValidData()) {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }

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

        // default values are set to invalid where necessary, so this is fine
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
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
            if (blackboard.playerToTicksSinceLastProbDeltaPosAssignment.find(treeThinker.csgoId) ==
                blackboard.playerToTicksSinceLastProbDeltaPosAssignment.end()) {
                blackboard.playerToTicksSinceLastProbDeltaPosAssignment[treeThinker.csgoId] = newDeltaPosTicks;
            }
            blackboard.playerToTicksSinceLastProbDeltaPosAssignment[treeThinker.csgoId]++;
            bool timeForNewDeltaPos =
                    blackboard.playerToTicksSinceLastProbDeltaPosAssignment.at(treeThinker.csgoId) >= newDeltaPosTicks ||
                    wasInEngagement;
            if (blackboard.playerToLastProbDeltaPosAssignment.find(treeThinker.csgoId) ==
                blackboard.playerToLastProbDeltaPosAssignment.end() || timeForNewDeltaPos) {
                blackboard.playerToLastProbDeltaPosAssignment[treeThinker.csgoId] =
                        {Vec3{INVALID_ID, INVALID_ID, INVALID_ID}, 0, 0, false};
            }
            PriorityDeltaPosAssignment & lastProbDeltaPosAssignment =
                    blackboard.playerToLastProbDeltaPosAssignment[treeThinker.csgoId];
            if (!lastProbDeltaPosAssignment.valid) {
                computeDeltaPosProbabilistic(state, curPriority, treeThinker.csgoId, modelNavData);
                lastProbDeltaPosAssignment = {curPriority.targetPos, curPriority.targetAreaId, modelNavData.deltaPosIndex, true};
                blackboard.playerToTicksSinceLastProbDeltaPosAssignment[treeThinker.csgoId] = 0;
                lastProbDeltaPosAssignment.valid = true;
            }
            else {
                curPriority.targetPos = lastProbDeltaPosAssignment.targetPos;
                curPriority.targetAreaId = lastProbDeltaPosAssignment.targetAreaId;
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
            /*
            // if CT defuser and in bombsite, then move to c4
            if (blackboard.isPlayerDefuser(treeThinker.csgoId) &&
                blackboard.navFile.get_place(curArea.m_place) == curOrder.waypoints.back().placeName) {
                curPriority.targetPos = state.getC4Pos();
                curPriority.targetAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();
            }
             */

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
