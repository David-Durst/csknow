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
                static_cast<double>(modelNavData.deltaXVal)/* * csknow::feature_store::delta_pos_grid_cell_dim)*/,
                static_cast<double>(modelNavData.deltaYVal)/* * csknow::feature_store::delta_pos_grid_cell_dim)*/,
                modelNavData.deltaZVal > 1 ? MAX_JUMP_HEIGHT : 0.
        };

        // if not jumping or falling, get nearest in 3d
        size_t navAboveBelowIndex = blackboard.navAboveBelow.posToIndex(curPriority.targetPos);
        if (modelNavData.deltaZVal == 1 || !blackboard.navAboveBelow.foundBelow[navAboveBelowIndex]) {
            curPriority.targetAreaId = blackboard.navAboveBelow.areaNearest[navAboveBelowIndex];
        }
        else {
            curPriority.targetAreaId = blackboard.navAboveBelow.areaBelow[navAboveBelowIndex];
        }

        /*
        curPriority.targetPos = Vec3{-2182.96875, 2191.69189453125, 1.5059620141983032} + Vec3{
                static_cast<double>(-1 * csknow::feature_store::delta_pos_grid_cell_dim),
                static_cast<double>(0 * csknow::feature_store::delta_pos_grid_cell_dim),
                0.
        };
         */


        /*
        curPriority.targetAreaId = blackboard.navFile
                .get_nearest_area_by_position_z_limit(vec3Conv(curPriority.targetPos), 2000.f, 2*MAX_JUMP_HEIGHT).get_id();
                */
        /*
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
         */

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
         */
        curPriority.targetPos = vec3tConv(blackboard.navFile.get_nearest_point_in_area(
                vec3Conv(curPriority.targetPos), blackboard.navFile.get_area_by_id_fast(curPriority.targetAreaId)));
        /*
        if (!(curPriority.targetPos == desiredPos)) {
            std::cout << curClient.name << " cur pos " << curClient.getFootPosForPlayer().toCSV() << " target pos " << curPriority.targetPos.toCSV() << " desired pos " << desiredPos.toCSV() << std::endl;
        }
         */

        modelNavData.nextArea = curPriority.targetAreaId;
        return desiredPos;
    }

    // 11 blend points as first one will be starting point. want 10 points over entire length
    constexpr int num_blend_points = 11;

    void ComputeModelNavAreaNode::checkClearPathToTargetPos(const ServerState::Client & curClient,
                                                            Priority & curPriority) {
        AreaId lastAreaId = INVALID_ID;
        // for each sample, verify that can walk from prior sample to next one
        // 1. base case - in the same nav mesh as in the prior one
        // 2. connected case - a nav mesh that shares an edge with the prior one
        //      a. since sampling at very fine level, this case shouldn't fail when 2 nav mesh cells are connected, but not at the location where a bot wants to cross (like rounding a corner)
        // 3. complex case - nav mesh that touches another navmesh
        //      c. this handles the case where navmesh cells form a 2x2 grid, and can't walk diagonally because those don't share an edge
        set<AreaId> badStartIds{4052};
        set<AreaId> badEndIds{7555};
        AreaId startAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curClient.getFootPosForPlayer())).get_id();
        AreaId endAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();
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
            for (const auto & conId : blackboard.mapMeshResult.connectionAreaIds[lastAreaIndex]) {
                if (conId == curAreaId) {
                    connectedNavAreas = true;
                    break;
                }
            }

            // check if nav areas touching
            AABB lastAreaRegion = blackboard.mapMeshResult.coordinate[lastAreaIndex];
            AABB curAreaRegion = blackboard.mapMeshResult.coordinate[curAreaIndex];
            // expand one area region a little for tolerance
            lastAreaRegion.min.z -= MAX_OBSTACLE_SIZE;
            lastAreaRegion.max.z += MAX_OBSTACLE_SIZE;
            bool touchingNavAreas = aabbOverlap(lastAreaRegion, curAreaRegion);

            if (!sameNavArea && !connectedNavAreas && !touchingNavAreas) {
                if (curClient.name == "Steel") {
                    std::cout << "huh" << std::endl;
                }
                std::cout << curClient.name << " not valid path" << std::endl;
                curPriority.directPathToLearnedTargetPos = false;
                if (lastAreaId == 8107 || lastAreaId == 4057 || lastAreaId == 4118 || lastAreaId == 4119 || lastAreaId == 4052) {
                    std::cout << "weird" << std::endl;
                }
                if (badStartIds.count(startAreaId) && badEndIds.count(endAreaId)) {
                    std::cout << "more weird" << std::endl;
                }
                return;
            }
            if (lastAreaId == 8107 && (curAreaId == 4138 || curAreaId == 4137 || curAreaId == 1414 || curAreaId == 7581)) {
                std::cout << "bad" << std::endl;
            }
            lastAreaId = curAreaId;
        }
        //set<AreaId> badStartIds{8107, 4118, 4119, 4057, 4052, 7557};
        //set<AreaId> badEndIds{7581, 4138, 4139, 1414};
        if (badStartIds.count(startAreaId) && badEndIds.count(endAreaId)) {
            std::cout << "another bad" << std::endl;
        }
        curPriority.directPathToLearnedTargetPos = true;
    }

    void ComputeModelNavAreaNode::computeDeltaPosProbabilistic(const ServerState & state, Priority & curPriority,
                                                               CSGOId csgoId, ModelNavData & modelNavData) {
        // compute area probabilities
        const csknow::inference_delta_pos::InferenceDeltaPosPlayerAtTickProbabilities & deltaPosProbabilities =
                blackboard.inferenceManager.playerToInferenceData.at(csgoId).deltaPosProbabilities;
        vector<float> probabilities = deltaPosProbabilities.radialVelProbabilities;
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
        // don't update learned pos while waiting for a path correction to finish
        // once target pos is valid, check if clear path before and after repathing for low latency
        //if (curPriority.directPathToLearnedTargetPos) {
        //    checkClearPathToTargetPos(curClient, curPriority);
        //}
        size_t deltaPosOption = 0;
        // / *
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
        /*
        if (modelNavData.radialVelIndex == 64 || modelNavData.radialVelIndex == 67 || modelNavData.radialVelIndex == 70 || modelNavData.radialVelIndex == 73 || modelNavData.radialVelIndex == 76 || modelNavData.radialVelIndex == 79 || modelNavData.radialVelIndex == 82) {
            std::cout << "hi" << std::endl;
        }
         */
        csknow::weapon_speed::MovementStatus movementStatus(static_cast<EngineWeaponId>(curClient.currentWeaponId),
                                                            curClient.isScoped, modelNavData.radialVelIndex);
        modelNavData.deltaXVal = movementStatus.vel.x;
        modelNavData.deltaYVal = movementStatus.vel.y;
        modelNavData.deltaZVal = movementStatus.zBin;
        curPriority.moveOptions.walk = movementStatus.statureOption == weapon_speed::StatureOptions::Walking;
        curPriority.moveOptions.crouch = movementStatus.statureOption == weapon_speed::StatureOptions::Ducking;

        tryDeltaPosTargetPos(state, curClient, curPriority, modelNavData);
        checkClearPathToTargetPos(curClient, curPriority);

        /*
        // this triggers is in b and model never seen defuse (so trying to go away) but c4 active so go wrong way
        if (modelNavData.deltaXVal < 0 && curClient.getFootPosForPlayer().x < curPriority.targetPos.x) {
            int x = 1;
            curPriority.targetPos = priorTargetPos;
            tryDeltaPosTargetPos(state, curClient, curPriority, modelNavData);
            (void) x;
        }
         */

        /*
         * this worked when deltaX/Y == 0 because mul by 2
         * this was for bad angled areas. I coudl try replacing it with taking nearest pos in area by 2d rather than 3d
         * distance, which would fix angled pos and work because all area don't overlap in z
         * NOT SURE IF NECESSARY NOW THAT ALLOWING CLOSER TO EDGE IF CLOSE IN COMPUTE PATH
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
        if (priorTargetAreaId == 6803 && curPriority.targetAreaId == 8802) {
            std::cout << "jumping" << std::endl;
        }
         */
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
                        {Vec3{INVALID_ID, INVALID_ID, INVALID_ID}, 0, 0, false, false, false};
            }
            PriorityDeltaPosAssignment & lastProbDeltaPosAssignment =
                    blackboard.playerToLastProbDeltaPosAssignment[treeThinker.csgoId];
            if (!lastProbDeltaPosAssignment.valid) {
                computeDeltaPosProbabilistic(state, curPriority, treeThinker.csgoId, modelNavData);
                lastProbDeltaPosAssignment = {curPriority.targetPos, curPriority.targetAreaId, modelNavData.radialVelIndex,
                                              curPriority.moveOptions.walk, curPriority.moveOptions.crouch, true};
                blackboard.playerToTicksSinceLastProbDeltaPosAssignment[treeThinker.csgoId] = 0;
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
