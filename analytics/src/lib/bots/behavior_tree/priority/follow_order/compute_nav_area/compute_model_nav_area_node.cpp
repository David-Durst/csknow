//
// Created by durst on 4/22/23.
//

#include "bots/analysis/learned_models.h"
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace follow::compute_nav_area {
    constexpr double max_place_distance_seconds = 5.;

    PlaceIndex ComputeModelNavAreaNode::computePlaceProbabilistic(const ServerState & state, const Order & curOrder,
                                                                  AreaId curAreaId, CSGOId csgoId) {
        const ServerState::Client & curClient = state.getClient(csgoId);
        // get cur place (closest place on order) and all places closer to objective if CT (offense)
        // include a places on order if T (on defense)
        set<PlaceIndex> waypointPlacesSet;
        for (const auto & waypoint : curOrder.waypoints) {
            waypointPlacesSet.insert(blackboard.distanceToPlaces.placeNameToIndex.at(waypoint.placeName));
        }

        PlaceIndex curPlace = 0;
        double minPlaceDistance = std::numeric_limits<double>::max();
        size_t curAreaIndex = blackboard.navFile.m_area_ids_to_indices.at(curAreaId);
        for (PlaceIndex placeIndex = 0; placeIndex < blackboard.distanceToPlaces.places.size(); placeIndex++) {
            double curPlaceDistance = blackboard.distanceToPlaces.getClosestDistance(curAreaIndex, placeIndex);
            if (curPlaceDistance < minPlaceDistance && waypointPlacesSet.find(placeIndex) != waypointPlacesSet.end()) {
                curPlace = placeIndex;
                minPlaceDistance = curPlaceDistance;
            }
        }

        string curPlaceName = blackboard.distanceToPlaces.places[curPlace];
        set<PlaceIndex> validPlaces;
        bool hitCurPlace = false;
        for (const auto & waypoint : curOrder.waypoints) {
            if (waypoint.placeName == curPlaceName) {
                hitCurPlace = true;
            }
            if (hitCurPlace || curClient.team == ENGINE_TEAM_T) {
                validPlaces.insert(blackboard.distanceToPlaces.placeNameToIndex.at(waypoint.placeName));
            }
        }

        /*
        // add all places that are within 5s of current location and not on order (aka can explore but can't go back)
        for (const auto & placeName : blackboard.distanceToPlaces.places) {
            if (secondsAwayAtMaxSpeed(blackboard.distanceToPlaces.getClosestDistance(curAreaId, placeName, blackboard.navFile)) <
                max_place_distance_seconds) {
                validPlaces.insert(blackboard.distanceToPlaces.placeNameToIndex.at(placeName));
            }
        }
         */

        // compute place probabilities
        vector<float> probabilities;
        vector<PlaceIndex> validPlacesVector;
        const csknow::inference_latent_place::InferencePlacePlayerAtTickProbabilities & placeProbabilities =
            blackboard.inferenceManager.playerToInferenceData.at(csgoId).placeProbabilities;
        for (PlaceIndex i = 0; i < placeProbabilities.placeProbabilities.size(); i++) {
            if (validPlaces.find(i) != validPlaces.end()) {
                probabilities.push_back(placeProbabilities.placeProbabilities[i]);
                validPlacesVector.push_back(i);
            }
        }
        // this should print for t's only
        /*
        if (curOrder.waypoints[0].type == WaypointType::C4) {
            std::cout << "valid places for " << csgoId << ": ";
            for (const auto validPlace : validPlaces) {
                std::cout << blackboard.distanceToPlaces.places[validPlace] << ", ";
            }
            std::cout << std::endl;
            std::cout << "cur place: " << curPlaceName << std::endl;
        }
         */

        // re-weight just for valid places
        double reweightFactor = 0.;
        for (size_t i = 0; i < probabilities.size(); i++) {
            reweightFactor += probabilities[i];
        }
        for (size_t i = 0; i < probabilities.size(); i++) {
            probabilities[i] *= 1/reweightFactor;
        }
        double probSample = blackboard.aggressionDis(blackboard.gen);
        double weightSoFar = 0.;
        PlaceIndex placeOption = 0;
        for (size_t i = 0; i < probabilities.size(); i++) {
            weightSoFar += probabilities[i];
            if (probSample < weightSoFar) {
                placeOption = validPlacesVector[i];
                break;
            }
        }
        return placeOption;
    }

    void ComputeModelNavAreaNode::computeAreaProbabilistic(Priority & curPriority, PlaceIndex nextPlace,
                                                           CSGOId csgoId) {
        // compute area probabilities
        const csknow::inference_latent_area::InferenceAreaPlayerAtTickProbabilities & areaGridProbabilities =
            blackboard.inferenceManager.playerToInferenceData.at(csgoId).areaProbabilities;
        vector<float> probabilities = areaGridProbabilities.areaProbabilities;

        // re-weight just because want to be certain sums to one
        double reweightFactor = 0.;
        for (size_t i = 0; i < probabilities.size(); i++) {
            reweightFactor += probabilities[i];
        }
        for (size_t i = 0; i < probabilities.size(); i++) {
            probabilities[i] *= 1/reweightFactor;
        }
        double probSample = blackboard.aggressionDis(blackboard.gen);
        double weightSoFar = 0.;
        size_t areaGridOption = 0;
        for (size_t i = 0; i < probabilities.size(); i++) {
            weightSoFar += probabilities[i];
            if (probSample < weightSoFar) {
                areaGridOption = i;
                break;
            }
        }

        // compute map grid to pos, and then pos to area
        // ok to pick bad area, as computePath in path node will pick a valid alternative (tree computes alternatives)
        const AABB & placeAABB = blackboard.distanceToPlaces.placeToAABB[blackboard.distanceToPlaces.places[nextPlace]];
        double deltaX = (placeAABB.max.x - placeAABB.min.x) / csknow::feature_store::area_grid_dim;
        double deltaY = (placeAABB.max.y - placeAABB.min.y) / csknow::feature_store::area_grid_dim;
        size_t xVal = areaGridOption % csknow::feature_store::area_grid_dim;
        size_t yVal = areaGridOption / csknow::feature_store::area_grid_dim;
        // add half so get center of each area grid
        // no need for z since doing 2d compare
        Vec3 areaGridPos = placeAABB.min + Vec3{
            (xVal + 0.5) * deltaX,
            (yVal + 0.5) * deltaY,
            0.
        };
        curPriority.targetAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(areaGridPos)).get_id();
        curPriority.targetPos = vec3tConv(blackboard.navFile.get_area_by_id_fast(curPriority.targetAreaId).get_center());
    }

    NodeState ComputeModelNavAreaNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (blackboard.inAnalysis || blackboard.inTest || !usePlaceAreaModelProbabilities) {
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
        bool wasInEngagement = curPriority.priorityType == PriorityType::Engagement;
        curPriority.priorityType = PriorityType::Order;
        curPriority.targetPlayer.playerId = INVALID_ID;
        curPriority.nonDangerAimArea = {};
        curPriority.moveOptions = {true, false, false};
        curPriority.shootOptions = ShootOptions::DontShoot;

        // if put in the model orders but not ready for this player, just stand still
        if (blackboard.inferenceManager.playerToInferenceData.find(treeThinker.csgoId) ==
            blackboard.inferenceManager.playerToInferenceData.end() ||
            !blackboard.inferenceManager.playerToInferenceData.at(treeThinker.csgoId).validData) {
            curPriority.targetPos = state.getClient(treeThinker.csgoId).getFootPosForPlayer();
            curPriority.targetAreaId = curAreaId;
        }
        else {
            if (blackboard.playerToTicksSinceLastProbPlaceAreaAssignment.find(treeThinker.csgoId) ==
                blackboard.playerToTicksSinceLastProbPlaceAreaAssignment.end()) {
                blackboard.playerToTicksSinceLastProbPlaceAreaAssignment[treeThinker.csgoId] = newPlaceAreaTicks;
            }
            blackboard.playerToTicksSinceLastProbPlaceAreaAssignment[treeThinker.csgoId]++;
            bool timeForNewPlaceArea =
                blackboard.playerToTicksSinceLastProbPlaceAreaAssignment.at(treeThinker.csgoId) >= newTargetTicks ||
                wasInEngagement;
            if (blackboard.playerToLastProbPlaceAreaAssignment.find(treeThinker.csgoId) ==
                blackboard.playerToLastProbPlaceAreaAssignment.end() || timeForNewPlaceArea) {
                blackboard.playerToLastProbPlaceAreaAssignment[treeThinker.csgoId] =
                    {Vec3{INVALID_ID, INVALID_ID, INVALID_ID}, 0, false};
            }
            PriorityPlaceAreaAssignment & lastProbPlaceAreaAssignment =
                blackboard.playerToLastProbPlaceAreaAssignment[treeThinker.csgoId];

            if (!lastProbPlaceAreaAssignment.valid) {
                PlaceIndex nextPlace = computePlaceProbabilistic(state, curOrder, curAreaId, treeThinker.csgoId);
                computeAreaProbabilistic(curPriority, nextPlace, treeThinker.csgoId);
                lastProbPlaceAreaAssignment = {curPriority.targetPos, curPriority.targetAreaId, true};
                blackboard.playerToTicksSinceLastProbPlaceAreaAssignment[treeThinker.csgoId] = 0;
            }
            else {
                curPriority.targetPos = lastProbPlaceAreaAssignment.targetPos;
                curPriority.targetAreaId = lastProbPlaceAreaAssignment.targetAreaId;
            }

            // if CT defuser and in bombsite, then move to c4
            if (blackboard.isPlayerDefuser(treeThinker.csgoId) &&
                blackboard.distanceToPlaces.places[curArea.m_place] == curOrder.waypoints.back().placeName) {
                curPriority.targetPos = state.getC4Pos();
                curPriority.targetAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();
            }

            // if in the target area (and not moving to c4), don't move
            if ((!blackboard.isPlayerDefuser(treeThinker.csgoId) || state.c4IsDefused) && curAreaId == curPriority.targetAreaId) {
                curPriority.moveOptions = {false, false, false};
            }
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
