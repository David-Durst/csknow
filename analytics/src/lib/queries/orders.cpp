//
// Created by durst on 2/19/23.
//

#include "queries/orders.h"

namespace csknow::orders {
    struct PossibleOrder {
        std::vector<PlaceIndex> placesVec;
        std::set<PlaceIndex> placesSet;
        std::vector<string> placesNames;

        PossibleOrder(const std::vector<PlaceIndex> & places, const std::vector<string> & placesNames) :
            placesVec(places), placesSet(placesVec.begin(), placesVec.end()), placesNames(placesNames) { }
    };

    // for each place, compute all places that are reachable in one area edge
    map<PlaceIndex, vector<PlaceIndex>> OrdersResult::computeConnectedPlaces() const {
        map<PlaceIndex, set<PlaceIndex>> result;
        for (PlaceIndex srcPlaceIndex = 0; srcPlaceIndex < distanceToPlacesResult.places.size(); srcPlaceIndex++) {
            const string & srcPlaceName = distanceToPlacesResult.places[srcPlaceIndex];
            for (const auto & srcAreaId : distanceToPlacesResult.placeToArea.at(srcPlaceName)) {
                int64_t srcAreaIndex = mapMeshResult.areaToInternalId.at(srcAreaId);
                for (size_t i = 0; i < mapMeshResult.connectionAreaIds[srcAreaIndex].size(); i++) {
                    int64_t destIndex = mapMeshResult.areaToInternalId.at(mapMeshResult.connectionAreaIds[srcAreaIndex][i]);
                    PlaceIndex dstPlaceIndex = distanceToPlacesResult.areaToPlace[destIndex];
                    // TODO: fix with ability to walk through invalid regions, for now, fixing one in bdoors/b on d2
                    if (destIndex == 1077) {
                        dstPlaceIndex = distanceToPlacesResult.placeNameToIndex.at("BombsiteB");
                    }
                    if (dstPlaceIndex != srcPlaceIndex && distanceToPlacesResult.placeValid(dstPlaceIndex)) {
                        result[srcPlaceIndex].insert(dstPlaceIndex);
                    }
                }
            }
        }

        map<PlaceIndex, vector<PlaceIndex>> vecResult;
        for (const auto & [placeIndex, connectedPlaceIndices] : result) {
            vecResult[placeIndex] = std::vector<PlaceIndex>(connectedPlaceIndices.begin(), connectedPlaceIndices.end());
        }
        return vecResult;
    }

    // need to do a DFS where allow re-exploration, just not cycles
    struct DFSOrderState {
        std::vector<PlaceIndex> curPath;
        std::vector<string> curPathNames;
        std::vector<std::vector<PlaceIndex>> branchOptions;
        std::vector<size_t> curBranchIndices;

        void pop_and_increment() {
            curPath.pop_back();
            curPathNames.pop_back();
            branchOptions.pop_back();
            curBranchIndices.pop_back();
            if (!curBranchIndices.empty()) {
                curBranchIndices.back()++;
            }
        }
    };

    std::vector<PossibleOrder> computeAllPossibleOrders(const DistanceToPlacesResult & distanceToPlacesResult,
            const map<PlaceIndex, vector<PlaceIndex>> & connectedPlaces, OrderLandmarks landmarks) {
        std::vector<PossibleOrder> result;

        DFSOrderState dfsOrderState{{landmarks.startPlace},
                                    {distanceToPlacesResult.places.at(landmarks.startPlace)},
                                    {connectedPlaces.at(landmarks.startPlace)}, {0}};

        // done if empty
        while (!dfsOrderState.branchOptions.empty()) {
            // done with current level of branch if hit all options
            if (dfsOrderState.curBranchIndices.back() == dfsOrderState.branchOptions.back().size()) {
                dfsOrderState.pop_and_increment();
            }
            // append path if it ends at goal
            else if (dfsOrderState.curPath.back() == landmarks.endPlace) {
                result.push_back(PossibleOrder(dfsOrderState.curPath, dfsOrderState.curPathNames));
                dfsOrderState.pop_and_increment();
            }
            // if not start, other objective, or cycle, then step in a layer
            else {
                PlaceIndex nextPlaceIndex = dfsOrderState.branchOptions.back()[dfsOrderState.curBranchIndices.back()];
                bool nextPlaceIsStart = nextPlaceIndex == landmarks.startPlace;
                bool nextPlaceIsOtherObjective = nextPlaceIndex == landmarks.otherObjective;
                bool cycle = std::find(dfsOrderState.curPath.begin(), dfsOrderState.curPath.end(), nextPlaceIndex) !=
                    dfsOrderState.curPath.end();
                // map mesh has some accidental duplicate names, like Ramp and ARamp
                bool invalidName = distanceToPlacesResult.places[nextPlaceIndex] == "Ramp";
                if (!nextPlaceIsStart && !nextPlaceIsOtherObjective && !cycle && !invalidName) {
                    dfsOrderState.curPath.push_back(nextPlaceIndex);
                    dfsOrderState.curPathNames.push_back(distanceToPlacesResult.places[nextPlaceIndex]);
                    dfsOrderState.branchOptions.push_back(connectedPlaces.at(nextPlaceIndex));
                    dfsOrderState.curBranchIndices.push_back(0);
                }
                else {
                    dfsOrderState.curBranchIndices.back()++;
                }
            }
        }

        return result;
    }

    std::vector<PossibleOrder> filterSupersetOrders(std::vector<PossibleOrder> & allPossibleOrders) {
        std::sort(allPossibleOrders.begin(), allPossibleOrders.end(),
                  [](const PossibleOrder & a, const PossibleOrder & b)
                  { return a.placesVec.size() > b.placesVec.size(); });
        std::vector<PossibleOrder> result;

        for (size_t possibleOrderIndex = 0; possibleOrderIndex < allPossibleOrders.size(); possibleOrderIndex++) {
            bool isSuperset = false;
            for (size_t smallerOrderIndex = possibleOrderIndex + 1; smallerOrderIndex < allPossibleOrders.size();
                 smallerOrderIndex++) {
                std::vector<PlaceIndex> intersection;
                std::set_intersection(
                    allPossibleOrders[smallerOrderIndex].placesSet.begin(),
                    allPossibleOrders[smallerOrderIndex].placesSet.end(),
                    allPossibleOrders[possibleOrderIndex].placesSet.begin(),
                    allPossibleOrders[possibleOrderIndex].placesSet.end(),
                    std::back_inserter(intersection));
                if (intersection.size() == allPossibleOrders[smallerOrderIndex].placesSet.size()) {
                    isSuperset = true;
                    break;
                }
            }

            if (!isSuperset) {
                result.push_back(allPossibleOrders[possibleOrderIndex]);
            }
        }

        return result;
    }

    std::set<PlaceIndex> OrdersResult::computePlacesVisibleFromDestination(OrderLandmarks landmarks) const {
        std::set<PlaceIndex> result;
        for (const auto & place : distanceToPlacesResult.places) {
            for (const auto & areaId : distanceToPlacesResult.placeToArea.at(place)) {
                if (visPoints.isVisiblePlace(areaId, distanceToPlacesResult.places[landmarks.endPlace],
                                             distanceToPlacesResult.placeToArea)) {
                    result.insert(distanceToPlacesResult.placeNameToIndex.at(place));
                    break;
                }
            }

        }
        return result;
    }

    // compute places that are unique in startOrderIndex relative to later ones and visible to destination
    std::vector<PlaceIndex> computeOrderRelativeUniqueVisiblePlaces(std::vector<PossibleOrder> & possibleOrders,
                                                                    std::set<PlaceIndex> & placesVisibleFromDestination,
                                                                    size_t startOrderIndex) {
        std::map<PlaceIndex, size_t> placeToNumPaths;
        for (size_t orderIndex = startOrderIndex; orderIndex < possibleOrders.size(); orderIndex++) {
            for (const auto & place : possibleOrders[orderIndex].placesVec) {
                if (placesVisibleFromDestination.find(place) != placesVisibleFromDestination.end()) {
                    if (placeToNumPaths.find(place) == placeToNumPaths.end()) {
                        placeToNumPaths[place] = 1;
                    }
                    else {
                        placeToNumPaths[place]++;
                    }
                }
            }
        }

        std::vector<PlaceIndex> result;
        for (const auto & placeIndex : possibleOrders[startOrderIndex].placesVec) {
            if (placeToNumPaths.find(placeIndex) != placeToNumPaths.end() && placeToNumPaths[placeIndex] == 1) {
                result.push_back(placeIndex);
            }
        }
        return result;
    }

    std::vector<PossibleOrder> filterNonUniqueOrders(std::vector<PossibleOrder> & nonSupersetOrders,
                                                     std::set<PlaceIndex> & placesVisibleFromDestination) {
        // this is done by filterSupersetOrders, but sorting again because cheap and defensive
        std::sort(nonSupersetOrders.begin(), nonSupersetOrders.end(),
                  [](const PossibleOrder & a, const PossibleOrder & b)
                  { return a.placesVec.size() > b.placesVec.size(); });
        std::vector<PossibleOrder> result;

        for (size_t orderIndex = 0; orderIndex < nonSupersetOrders.size(); orderIndex++) {
            std::vector<PlaceIndex> orderUniquePlaces =
                computeOrderRelativeUniqueVisiblePlaces(nonSupersetOrders, placesVisibleFromDestination,
                                                        orderIndex);
            if (!orderUniquePlaces.empty()) {
                result.push_back(nonSupersetOrders[orderIndex]);
            }
        }
        return result;
    }

    void OrdersResult::runQuery() {
        map<PlaceIndex, vector<PlaceIndex>> connectedPlaces = computeConnectedPlaces();

        OrderLandmarks aLandmarks{distanceToPlacesResult.placeNameToIndex.at("TSpawn"),
                                  distanceToPlacesResult.placeNameToIndex.at("BombsiteA"),
                                  distanceToPlacesResult.placeNameToIndex.at("BombsiteB")},
                       bLandmarks{distanceToPlacesResult.placeNameToIndex.at("TSpawn"),
                                  distanceToPlacesResult.placeNameToIndex.at("BombsiteB"),
                                  distanceToPlacesResult.placeNameToIndex.at("BombsiteA")};

        // step 1: any possible sequence of places that don't repeat and don't hit start/other objective
        std::vector<PossibleOrder> aAllPossibleOrders =
                computeAllPossibleOrders(distanceToPlacesResult, connectedPlaces, aLandmarks),
                bAllPossibleOrders = computeAllPossibleOrders(distanceToPlacesResult, connectedPlaces, bLandmarks);

        // step 2: sort longest to shortest number of places, cull longest if anything shorter is strict subset with same
        // non-end last
        // need to compare non-start first and non-end last as want to remove order like pit in long A path
        // while keeping unique endings like B window
        std::vector<PossibleOrder> aNonSupersetOrders = filterSupersetOrders(aAllPossibleOrders),
            bNonSupersetOrders = filterSupersetOrders(bAllPossibleOrders);

        // step 3: longest to shortest number of places, cull all paths without a unique place that can see destination
        // this removes orders like
        //  1. cat to under A on d2
        //  2. paths through underpass on mirage
        //  3. mid to dumpster to B on cache
        // while keeping
        //  1. B window on d2
        //  2. CT spawn to A on d2
        //  3. mid to conn to a on mirage
        //  4. tboxes to B on cache
        // only doing retakes, not looking at early match exploration. important thing is it canonicalizes to one
        // order to B on cache through B halls
        // NOTE: unique relative to smaller paths, as non-unique areas will always be unique to one (smallest)
        std::set<PlaceIndex> aPlacesVisibleFromDestination = computePlacesVisibleFromDestination(aLandmarks),
            bPlacesVisibleFromDestination = computePlacesVisibleFromDestination(bLandmarks);
        std::vector<PossibleOrder>
            aUniqueEndingOrders = filterNonUniqueOrders(aNonSupersetOrders, aPlacesVisibleFromDestination),
            bUniqueEndingOrders = filterNonUniqueOrders(bNonSupersetOrders, bPlacesVisibleFromDestination);

        for (const auto & aOrder : aUniqueEndingOrders) {
            orders.push_back({aOrder.placesVec, OrderType::AOrder});
        }
        for (const auto & bOrder : bUniqueEndingOrders) {
            orders.push_back({bOrder.placesVec, OrderType::BOrder});
        }
    }
}