//
// Created by durst on 4/22/23.
//

#include "bots/analysis/learned_models.h"
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace follow::compute_nav_area {
    constexpr double max_place_distance_seconds = 5.;

    PlaceIndex ComputeModelNavAreaNode::computePlaceProbabilistic(const ServerState & state, const Order & curOrder,
                                                                  AreaId curAreaId, CSGOId csgoId,
                                                                  ModelNavData & modelNavData) {
        const ServerState::Client & curClient = state.getClient(csgoId);
        const csknow::inference_latent_place::InferencePlacePlayerAtTickProbabilities & placeProbabilities =
            blackboard.inferenceManager.playerToInferenceData.at(csgoId).placeProbabilities;
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
        vector<PlaceIndex> validPlacesVectorOrderedOrder;
        bool hitCurPlace = false;
        map<PlaceIndex, size_t> placeIndexToWaypointIndex;
        size_t nearestWaypoint = 0;
        double nearestWaypointDistance = std::numeric_limits<double>::max();
        PlaceIndex lastPlaceIndex = blackboard.distanceToPlaces.placeNameToIndex.at(curOrder.waypoints.back().placeName);
        for (size_t i = 0; i < curOrder.waypoints.size(); i++) {
            const Waypoint & waypoint = curOrder.waypoints[i];
            PlaceIndex placeIndex = blackboard.distanceToPlaces.placeNameToIndex.at(waypoint.placeName);
            placeIndexToWaypointIndex[placeIndex] = i;
            if (waypoint.placeName == curPlaceName) {
                hitCurPlace = true;
            }
            if ((curClient.team == ENGINE_TEAM_CT && hitCurPlace) ||
                (curClient.team == ENGINE_TEAM_T &&
                blackboard.placesVisibleFromDestination.find(placeIndex) != blackboard.placesVisibleFromDestination.end())) {
                validPlaces.insert(placeIndex);
                validPlacesVectorOrderedOrder.push_back(placeIndex);
                modelNavData.orderPlaceOptions.push_back(waypoint.placeName);
                modelNavData.orderPlaceProbs.push_back(placeProbabilities.placeProbabilities[placeIndex]);
            }
            int64_t closestAreaIndexInPlace = blackboard.distanceToPlaces.getClosestArea(curAreaIndex, placeIndex);
            double newWaypointDistance = blackboard.reachability.getDistance(curAreaIndex, closestAreaIndexInPlace) +
                blackboard.distanceToPlaces.getClosestDistance(closestAreaIndexInPlace, lastPlaceIndex);
            if (newWaypointDistance < nearestWaypointDistance) {
                nearestWaypoint = i;
                nearestWaypointDistance = newWaypointDistance;
            }
        }
        if (!hitCurPlace && curClient.team == ENGINE_TEAM_CT) {
            const Waypoint & waypoint = curOrder.waypoints[nearestWaypoint];
            PlaceIndex placeIndex = blackboard.distanceToPlaces.placeNameToIndex.at(waypoint.placeName);
            validPlaces.insert(placeIndex);
            validPlacesVectorOrderedOrder.push_back(placeIndex);
            modelNavData.orderPlaceOptions.push_back(waypoint.placeName);
            modelNavData.orderPlaceProbs.push_back(placeProbabilities.placeProbabilities[placeIndex]);
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
        float curPlaceProbability = 0.;
        vector<float> probabilities;
        vector<PlaceIndex> validPlacesOrderedProbs;
        for (PlaceIndex i = 0; i < placeProbabilities.placeProbabilities.size(); i++) {
            if (validPlaces.find(i) != validPlaces.end()) {
                probabilities.push_back(placeProbabilities.placeProbabilities[i]);
                validPlacesOrderedProbs.push_back(i);
            }
            if (i == curPlace) {
                curPlaceProbability = placeProbabilities.placeProbabilities[i];
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
        // if CT (offense), give all non-cur weight to next place as on rails
        // if T, distribute evenly as less proscriptive
        PlaceIndex placeOption = 0;
        if (curClient.team == ENGINE_TEAM_CT) {
            double probSample = blackboard.aggressionDis(blackboard.gen);
            //std::cout << "csgoid " << csgoId << " probSample " << probSample << " curPlaceProbability " << curPlaceProbability << std::endl;
            if (probSample < curPlaceProbability) {
                placeOption = curPlace;
            }
            else {
                // if not cur place, next place (unless only 1 option as at end)
                if (validPlacesVectorOrderedOrder.size() == 1) {
                    placeOption = validPlacesVectorOrderedOrder[0];
                }
                else {
                    placeOption = validPlacesVectorOrderedOrder[1];
                }
            }
        }
        else {
            double reweightFactor = 0.;
            for (size_t i = 0; i < probabilities.size(); i++) {
                reweightFactor += probabilities[i];
            }
            for (size_t i = 0; i < probabilities.size(); i++) {
                probabilities[i] *= 1/reweightFactor;
            }
            double probSample = blackboard.aggressionDis(blackboard.gen);
            double weightSoFar = 0.;
            for (size_t i = 0; i < probabilities.size(); i++) {
                weightSoFar += probabilities[i];
                if (probSample < weightSoFar) {
                    placeOption = validPlacesOrderedProbs[i];
                    break;
                }
            }
        }

        // if cur place isn't next place and same order, terminate early
        if (blackboard.playerToModelNavData.find(csgoId) != blackboard.playerToModelNavData.end()) {
            const ModelNavData & oldModelNavData = blackboard.playerToModelNavData.at(csgoId);
            if (oldModelNavData.nextPlace != curPlaceName) {
                placeOption = oldModelNavData.nextPlaceIndex;
            }
        }

        // if T and new to order, just get to closet place on order before going anywhere else
        if (blackboard.playerToModelNavData.find(csgoId) == blackboard.playerToModelNavData.end() &&
            curClient.team == ENGINE_TEAM_T) {
            placeOption = curPlace;
        }

        // if cur place is bombsite and CT defuser, then move to c4
        if (blackboard.isPlayerDefuser(csgoId) && (curPlaceName == "BombsiteA" || curPlaceName == "BombsiteB")) {
            placeOption = validPlacesVectorOrderedOrder.back();
        }

        modelNavData.curPlace = curPlaceName;
        modelNavData.nextPlace = blackboard.distanceToPlaces.places[placeOption];
        modelNavData.nextPlaceIndex = placeOption;
        blackboard.strategy.playerToWaypointIndex[csgoId] = placeIndexToWaypointIndex[curPlace];
        return placeOption;
    }

    void ComputeModelNavAreaNode::computeAreaProbabilistic(const ServerState & state, Priority & curPriority,
                                                           PlaceIndex nextPlace, CSGOId csgoId,
                                                           ModelNavData & modelNavData) {
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

        curPriority.targetAreaId = blackboard.navFile
            .get_nearest_area_by_position_in_place(vec3Conv(areaGridPos), modelNavData.nextPlaceIndex).get_id();
        if (blackboard.removedAreas.find(curPriority.targetAreaId) != blackboard.removedAreas.end()) {
            curPriority.targetAreaId = blackboard.removedAreaAlternatives[curPriority.targetAreaId];
        }
        // if same next place and not at old next area, keep using that area
        if (blackboard.playerToModelNavData.find(csgoId) != blackboard.playerToModelNavData.end()) {
            const ModelNavData & oldModelNavData = blackboard.playerToModelNavData.at(csgoId);
            AreaId curAreaId = blackboard.navFile
                .get_nearest_area_by_position(vec3Conv(state.getClient(csgoId).getFootPosForPlayer())).get_id();
            if (oldModelNavData.nextPlace == modelNavData.nextPlace && oldModelNavData.nextArea != curAreaId) {
                curPriority.targetAreaId = oldModelNavData.nextArea;
            }
        }
        curPriority.targetPos = vec3tConv(blackboard.navFile.get_area_by_id_fast(curPriority.targetAreaId).get_center());
        modelNavData.nextArea = curPriority.targetAreaId;
    }

    NodeState ComputeModelNavAreaNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        if (blackboard.inAnalysis || blackboard.inTest || !getPlaceAreaModelProbabilities(curClient.team) ||
            !blackboard.inferenceManager.haveValidData()) {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }
        ModelNavData modelNavData;

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
                PlaceIndex nextPlace = computePlaceProbabilistic(state, curOrder, curAreaId, treeThinker.csgoId,
                                                                 modelNavData);
                computeAreaProbabilistic(state, curPriority, nextPlace, treeThinker.csgoId, modelNavData);
                lastProbPlaceAreaAssignment = {curPriority.targetPos, curPriority.targetAreaId, true};
                blackboard.playerToTicksSinceLastProbPlaceAreaAssignment[treeThinker.csgoId] = 0;
                blackboard.playerToModelNavData[treeThinker.csgoId] = modelNavData;
            }
            else {
                curPriority.targetPos = lastProbPlaceAreaAssignment.targetPos;
                curPriority.targetAreaId = lastProbPlaceAreaAssignment.targetAreaId;
            }

            // if CT defuser and in bombsite, then move to c4
            if (blackboard.isPlayerDefuser(treeThinker.csgoId) &&
                blackboard.navFile.get_place(curArea.m_place) == curOrder.waypoints.back().placeName) {
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
