//
// Created by durst on 5/2/22.
//

#ifndef CSKNOW_STRATEGY_DATA_H
#define CSKNOW_STRATEGY_DATA_H
#include "bots/load_save_bot_data.h"
#include "navmesh/nav_file.h"
#include "geometryNavConversions.h"
#include "queries/distance_to_places.h"
#include <sstream>
#include <algorithm>
using std::stringstream;
using std::map;

enum class ExecuteStatus {
    Setup,
    Ready,
    Executing
};

enum class WaypointType {
    NavPlace,
    NavAreas, // sometimes want to force a collection of areas without a name
    ChokePlace,
    ChokeAreas,
    HoldPlace,
    HoldAreas,
    C4,
    NUM_WAYPOINTS
};

struct Waypoint {
    WaypointType type;

    // use placeName if type of NavPlace, playerId if type of is player, placeName for site if c4
    // not using union because that prevented automatic destructor definition, and this is trivial amount of extra data
    // if placeName and areaIds are set, areaIds and invalid areas in place
    string placeName;
    string customAreasName;
    vector<AreaId> areaIds;
    bool aggresiveDefense;
    CSGOId playerId;
};

typedef vector<Waypoint> Waypoints;
/**
 * Order specifies how to to navigate the map as part of a strategy
 */
struct Order {
    Waypoints waypoints;
    vector<size_t> holdIndices;
    size_t aggressiveChokeIndex, passiveChokeIndex;
    // multiple players can watch one choke point, one player in a hold point
    map<CSGOId, size_t> playerToHoldIndex;
    map<size_t, AreaId> holdIndexToHoldAreaId, holdIndexToDangerAreaId;
    // what about chains of operations (like switching once plant happens)?

    void computeIndices(const nav_mesh::nav_file & navFile, const ReachableResult & reachability,
                        const VisPoints & visPoints, const DistanceToPlacesResult & distanceToPlacesResult) {
        holdIndices.clear();
        playerToHoldIndex.clear();
        for (size_t i = 0; i < waypoints.size(); i++) {
            if (waypoints[i].type == WaypointType::ChokePlace || waypoints[i].type == WaypointType::ChokeAreas) {
                if (waypoints[i].aggresiveDefense) {
                    aggressiveChokeIndex = i;
                }
                else {
                    passiveChokeIndex = i;
                }
            }
            else if (waypoints[i].type == WaypointType::HoldPlace || waypoints[i].type == WaypointType::HoldAreas) {
                holdIndices.push_back(i);
            }
        }

        for (const auto & holdIndex : holdIndices) {
            setClosestAreaToHideVisibleToChoke(holdIndex, navFile, reachability, visPoints, distanceToPlacesResult);
        }
    }

    double getDistance(AreaId srcAreaId, const vector<AreaId> & dstAreaIds, const nav_mesh::nav_file & navFile,
                       const ReachableResult & reachability) const {
        double minDistance = std::numeric_limits<double>::max();
        for (const auto & dstAreaId : dstAreaIds) {
            minDistance = std::min(minDistance, reachability.getDistance(srcAreaId, dstAreaId, navFile));
        }
        return minDistance;
    }

    double getDistance(AreaId srcAreaId, const Waypoint & waypoint, const nav_mesh::nav_file & navFile,
                       const ReachableResult & reachability, const DistanceToPlacesResult & distanceToPlacesResult) const {
        switch (waypoint.type) {
            case WaypointType::NavPlace:
                return distanceToPlacesResult.getDistance(srcAreaId, waypoint.placeName, navFile);
            case WaypointType::NavAreas:
                return getDistance(srcAreaId, waypoint.areaIds, navFile, reachability);
            case WaypointType::ChokePlace:
                return distanceToPlacesResult.getDistance(srcAreaId, waypoint.placeName, navFile);
            case WaypointType::ChokeAreas:
                return getDistance(srcAreaId, waypoint.areaIds, navFile, reachability);
            case WaypointType::HoldPlace:
                return distanceToPlacesResult.getDistance(srcAreaId, waypoint.placeName, navFile);
            case WaypointType::HoldAreas:
                return getDistance(srcAreaId, waypoint.areaIds, navFile, reachability);
            case WaypointType::C4:
                return distanceToPlacesResult.getDistance(srcAreaId, waypoint.placeName, navFile);
            default:
                throw std::runtime_error("invalid waypoint type for getting distance");
        }
    }

    std::optional<AreaId> isVisible(AreaId srcAreaId, const vector<AreaId> & dstAreaIds, const nav_mesh::nav_file & navFile,
                       const VisPoints & visPoints, bool farthest = false) const {
        bool foundVisibleArea = false;
        double maxDistance = -1 * std::numeric_limits<double>::max();
        AreaId farthestAreaId;
        for (const auto & dstAreaId : dstAreaIds) {
            if (visPoints.isVisibleAreaId(srcAreaId, dstAreaId)) {
                if (farthest) {
                    foundVisibleArea = true;
                    double newDistance = computeDistance(srcAreaId, dstAreaId, navFile);
                    if (newDistance > maxDistance) {
                        maxDistance = newDistance;
                        farthestAreaId = dstAreaId;
                    }
                }
                else {
                    return dstAreaId;
                }
            }
        }
        if (foundVisibleArea) {
            return farthestAreaId;
        }
        else {
            return {};
        }
    }

    std::optional<AreaId> isVisible(AreaId srcAreaId, const Waypoint & waypoint, const nav_mesh::nav_file & navFile,
                       const VisPoints & visPoints, const DistanceToPlacesResult & distanceToPlacesResult, bool farthest = false) const {
        switch (waypoint.type) {
            case WaypointType::NavPlace:
                return isVisible(srcAreaId, distanceToPlacesResult.placeToArea.find(waypoint.placeName)->second,
                                 navFile, visPoints, farthest);
            case WaypointType::NavAreas:
                return isVisible(srcAreaId, waypoint.areaIds, navFile, visPoints, farthest);
            case WaypointType::ChokePlace:
                return isVisible(srcAreaId, distanceToPlacesResult.placeToArea.find(waypoint.placeName)->second,
                                 navFile, visPoints, farthest);
            case WaypointType::ChokeAreas:
                return isVisible(srcAreaId, waypoint.areaIds, navFile, visPoints, farthest);
            case WaypointType::HoldPlace:
                return isVisible(srcAreaId, distanceToPlacesResult.placeToArea.find(waypoint.placeName)->second,
                                 navFile, visPoints, farthest);
            case WaypointType::HoldAreas:
                return isVisible(srcAreaId, waypoint.areaIds, navFile, visPoints, farthest);
            case WaypointType::C4:
                return isVisible(srcAreaId, distanceToPlacesResult.placeToArea.find(waypoint.placeName)->second,
                                 navFile, visPoints, farthest);
            default:
                throw std::runtime_error("invalid waypoint type for getting visibility");
        }
    }


    void setClosestAreaToHideVisibleToChoke(size_t waypointIndex, const nav_mesh::nav_file & navFile,
                                              const ReachableResult & reachability, const VisPoints & visPoints,
                                              const DistanceToPlacesResult & distanceToPlacesResult) {
        const Waypoint & waypoint = waypoints[waypointIndex];
        const Waypoint & chokeWaypoint = waypoints[waypoint.aggresiveDefense ? aggressiveChokeIndex : passiveChokeIndex];
        set<AreaId> invalidAreaIds;
        if (waypoint.type == WaypointType::HoldPlace) {
            invalidAreaIds.insert(waypoint.areaIds.begin(), waypoint.areaIds.end());
        }
        struct AreaOption {
            AreaId holdAreaId, chokeAreaId;
            double holdDistance;
            double chokeDistance;
        };
        vector<AreaOption> options;
        for (size_t areaIndex = 0; areaIndex < navFile.m_areas.size(); areaIndex++) {
            AreaId areaId = navFile.m_areas[areaIndex].get_id();
            if (invalidAreaIds.find(areaId) != invalidAreaIds.end()) {
                continue;
            }
            double newHoldDistance = getDistance(areaId, waypoint, navFile,
                                             reachability, distanceToPlacesResult);
            const auto optionalChokeAreaId = isVisible(areaId, chokeWaypoint, navFile,
                                                       visPoints, distanceToPlacesResult, true);
            if (newHoldDistance != NOT_CLOSEST_DISTANCE && optionalChokeAreaId) {
                // 3D distance (not walking distance) for visibility
                double newChokeDistance = computeDistance(areaId, optionalChokeAreaId.value(), navFile);
                options.push_back({areaId, optionalChokeAreaId.value(), newHoldDistance, newChokeDistance});
            }
        }
        std::sort(options.begin(), options.end(), [](const AreaOption & a, const AreaOption & b) {
            return a.holdDistance < b.holdDistance || (a.holdDistance == b.holdDistance && a.chokeDistance > b.chokeDistance);
        });
        holdIndexToHoldAreaId[waypointIndex] = options.front().holdAreaId;
        holdIndexToDangerAreaId[waypointIndex] = options.front().chokeAreaId;
    }

    void print(const vector<CSGOId> followers, const map<CSGOId, int64_t> & playerToWaypointIndex,
               const ServerState & state, const nav_mesh::nav_file & navFile,
               const map<CSGOId, int32_t> & playerToEntryIndex, const map<CSGOId, ExecuteStatus> & playerToExecuteStatus,
               size_t orderIndex, TeamId team, vector<string> & result) const {
        stringstream waypointsStream;
        if (team == ENGINE_TEAM_T) {
            waypointsStream << "T ";
        }
        else if (team == ENGINE_TEAM_CT) {
            waypointsStream << "CT ";
        }
        waypointsStream << orderIndex << " waypoints: [";
        for (const auto & waypoint : waypoints) {
            string typeString;
            string dataString;
            switch (waypoint.type) {
                case WaypointType::NavPlace:
                    typeString = "NavPlace";
                    dataString = waypoint.placeName;
                    break;
                case WaypointType::NavAreas:
                    typeString = "NavAreas";
                    dataString = waypoint.customAreasName;
                    break;
                case WaypointType::ChokePlace:
                    typeString = "ChokePlace";
                    dataString = waypoint.placeName;
                    break;
                case WaypointType::ChokeAreas:
                    typeString = "ChokeAreas";
                    dataString = waypoint.customAreasName;
                    break;
                case WaypointType::HoldPlace:
                    typeString = "HoldPlace";
                    dataString = waypoint.placeName;
                    break;
                case WaypointType::HoldAreas:
                    typeString = "HoldAreas";
                    dataString = waypoint.customAreasName;
                    break;
                case WaypointType::C4:
                    typeString = "C4";
                    dataString = waypoint.placeName;
                    break;
                default:
                    typeString = "INVALID_TYPE";
            }
            waypointsStream << "(" << typeString << "," << dataString << "); ";
        }
        waypointsStream << "]";
        result.push_back(waypointsStream.str());

        stringstream followersStream;
        if (team == ENGINE_TEAM_T) {
            followersStream << "T ";
        }
        else if (team == ENGINE_TEAM_CT) {
            followersStream << "CT ";
        }
        followersStream << orderIndex << " followers: <follower, waypoint index, entry index> : [";
        for (const auto & follower : followers) {
            followersStream << "<" << state.getPlayerString(follower) << ", "
                << playerToWaypointIndex.find(follower)->second << ", "
                << playerToEntryIndex.find(follower)->second;
            if (team == ENGINE_TEAM_CT && playerToExecuteStatus.find(follower) != playerToExecuteStatus.end()) {
                followersStream << ", ";
                ExecuteStatus status = playerToExecuteStatus.find(follower)->second;
                switch (status) {
                    case ExecuteStatus::Setup:
                        followersStream << "Setup";
                        break;
                    case ExecuteStatus::Ready:
                        followersStream << "Ready";
                        break;
                    case ExecuteStatus::Executing:
                        followersStream << "Executing";
                        break;
                    default:
                        followersStream << "INVALID";
                }
            }
            followersStream << ">; ";
        }
        followersStream << "]";
        result.push_back(followersStream.str());
    }
};

struct OrderId {
    TeamId team;
    int64_t index;
};

static bool operator<(const OrderId& a, const OrderId& b) {
    return a.team < b.team || (a.team == b.team && a.index < b.index);
}

class Strategy {
    vector<Order> tOrders, ctOrders;
    map<CSGOId, OrderId> playerToOrder;
    map<OrderId, vector<CSGOId>> orderToPlayers;
    map<CSGOId, ExecuteStatus> playerToExecuteStatus;

    // shouldn't need private suffix, but operator overlading isn't working for me for public op and private op
    Order & getOrderPrivate(OrderId orderId) {
        if (orderId.team == ENGINE_TEAM_T) {
            return tOrders[orderId.index];
        }
        else if (orderId.team == ENGINE_TEAM_CT) {
            return ctOrders[orderId.index];
        }
        throw std::runtime_error( "getOrderPrivate bad order id" );
    }

public:
    map<CSGOId, int64_t> playerToWaypointIndex;
    map<CSGOId, int32_t> playerToEntryIndex;

    void clear() {
        tOrders.clear();
        ctOrders.clear();
        playerToOrder.clear();
        orderToPlayers.clear();
        playerToWaypointIndex.clear();
        playerToEntryIndex.clear();
        playerToExecuteStatus.clear();
    }

    OrderId getOrderIdForPlayer(CSGOId playerId) const {
        if (playerToOrder.find(playerId) != playerToOrder.end()) {
            return playerToOrder.find(playerId)->second;
        }
        else {
            throw std::runtime_error( "getOrderForPlayer bad player id" );
        }
    }

    const vector<CSGOId> & getOrderFollowers(OrderId orderId) const {
        if (orderToPlayers.find(orderId) != orderToPlayers.end()) {
            return orderToPlayers.find(orderId)->second;
        }
        else {
            throw std::runtime_error( "getOrderFollowers bad order id" );
        }
    }

    const Order & getOrder(OrderId orderId) const {
        if (orderId.team == ENGINE_TEAM_T) {
            return tOrders[orderId.index];
        }
        else if (orderId.team == ENGINE_TEAM_CT) {
            return ctOrders[orderId.index];
        }
        throw std::runtime_error( "getOrder bad order id" );
    }

    const Order & getOrderForPlayer(CSGOId playerId) const {
        return getOrder(getOrderIdForPlayer(playerId));
    }

    const vector<OrderId> getOrderIds(bool includeT = true, bool includeCT = true) const {
        vector<OrderId> result;
        if (includeT) {
            for (size_t orderIndex = 0; orderIndex < tOrders.size(); orderIndex++) {
                result.push_back({ENGINE_TEAM_T, static_cast<int64_t>(orderIndex)});
            }
        }

        if (includeCT) {
            for (size_t orderIndex = 0; orderIndex < ctOrders.size(); orderIndex++) {
                result.push_back({ENGINE_TEAM_CT, static_cast<int64_t>(orderIndex)});
            }
        }
        return result;
    }

    OrderId addOrder(TeamId team, Order order, const nav_mesh::nav_file & navFile, const ReachableResult & reachability,
                     const VisPoints & visPoints, const DistanceToPlacesResult & distanceToPlacesResult) {
        order.computeIndices(navFile, reachability, visPoints, distanceToPlacesResult);
        OrderId orderId = {INVALID_ID, INVALID_ID};
        if (team == ENGINE_TEAM_T) {
            orderId = {ENGINE_TEAM_T, static_cast<int64_t>(tOrders.size())};
            tOrders.push_back(order);
        }
        else if (team == ENGINE_TEAM_CT) {
            orderId = {ENGINE_TEAM_CT, static_cast<int64_t>(ctOrders.size())};
            ctOrders.push_back(order);
        }
        orderToPlayers[orderId] = {};
        return orderId;
    }

    void assignPlayerToOrder(CSGOId playerId, OrderId orderId) {
        // remove player from existing order if already assigned
        if (playerToOrder.find(playerId) != playerToOrder.end()) {
            for (const auto & orderId : getOrderIds()) {
                vector<CSGOId> & followers = orderToPlayers[orderId];
                followers.erase(std::remove_if(followers.begin(), followers.end(),
                                               [playerId](CSGOId id) { return id == playerId; }), followers.end());
            }
        }
        playerToWaypointIndex[playerId] = 0;
        playerToOrder[playerId] = orderId;
        orderToPlayers[orderId].push_back(playerId);
    }

    double getDistance(AreaId srcAreaId, OrderId orderId, size_t waypointIndex, const nav_mesh::nav_file & navFile,
                       const ReachableResult & reachability, const DistanceToPlacesResult & distanceToPlacesResult) {
        const Order & order = getOrder(orderId);
        return order.getDistance(srcAreaId, order.waypoints[waypointIndex], navFile, reachability, distanceToPlacesResult);
    }

    void assignPlayerToHoldIndex(CSGOId playerId, OrderId orderId, size_t holdIndex) {
        for (Order & order : ctOrders) {
            order.playerToHoldIndex.erase(playerId);
        }
        for (Order & order : tOrders) {
            order.playerToHoldIndex.erase(playerId);
        }
        getOrderPrivate(orderId).playerToHoldIndex[playerId] = holdIndex;
    }

    vector<OrderId> getOrdersNotAssignedPlayers(TeamId team) const {
        vector<OrderId> result;
        const vector<Order> & orders = team == ENGINE_TEAM_T ? tOrders : ctOrders;
        for (size_t i = 0; i < orders.size(); i++) {
            if (orderToPlayers.find({team, static_cast<int64_t>(i)}) == orderToPlayers.end() ||
                orderToPlayers.find({team, static_cast<int64_t>(i)})->second.empty()) {
                result.push_back({team, static_cast<int64_t>(i)});
            }
        }
        return result;

    }

    int64_t maxTeamWaypoint(const ServerState & state, TeamId team) {
        int64_t result = INVALID_ID;
        for (const auto & csgoId : state.getPlayersOnTeam(team)) {
            if (state.getClient(csgoId).team == team && playerToWaypointIndex[csgoId] > result) {
                result = playerToWaypointIndex[csgoId];
            }
        }
        return result;
    }

    void playerSetup(CSGOId playerId) {
        // only allow swapping between ready and setup, once executing it's game time
        if (playerToExecuteStatus.find(playerId) == playerToExecuteStatus.end() ||
            playerToExecuteStatus.find(playerId)->second == ExecuteStatus::Ready) {
            playerToExecuteStatus[playerId] = ExecuteStatus::Setup;
        }
    }

    void playerReady(CSGOId playerId) {
        // only allow swapping between ready and setup, once executing it's game time
        if (playerToExecuteStatus.find(playerId) == playerToExecuteStatus.end() ||
            playerToExecuteStatus.find(playerId)->second == ExecuteStatus::Setup) {
            playerToExecuteStatus[playerId] = ExecuteStatus::Ready;
        }
    }

    void playerExecuting(CSGOId playerId) {
        playerToExecuteStatus[playerId] = ExecuteStatus::Executing;
    }

    bool playerFinishedSetup(CSGOId playerId) {
        return playerToExecuteStatus.find(playerId) != playerToExecuteStatus.end() &&
            playerToExecuteStatus.find(playerId)->second != ExecuteStatus::Setup;
    }

    bool isPlayerReady(CSGOId playerId) {
        return playerToExecuteStatus.find(playerId) != playerToExecuteStatus.end() &&
               playerToExecuteStatus.find(playerId)->second == ExecuteStatus::Ready;
    }

    bool isPlayerExecuting(CSGOId playerId) {
        return playerToExecuteStatus.find(playerId) != playerToExecuteStatus.end() &&
               playerToExecuteStatus.find(playerId)->second == ExecuteStatus::Executing;
    }

    vector<string> print(const ServerState & state, const nav_mesh::nav_file & navFile) const {
        vector<string> result;

        vector<TeamId> teams{ENGINE_TEAM_T, ENGINE_TEAM_CT};
        vector<vector<Order>> bothTeamOrders{tOrders, ctOrders};

        for (size_t teamIndex = 0; teamIndex < teams.size(); teamIndex++) {
            for (size_t orderIndex = 0; orderIndex < bothTeamOrders[teamIndex].size(); orderIndex++) {
                bothTeamOrders[teamIndex][orderIndex].print(orderToPlayers.find({teams[teamIndex], static_cast<int64_t>(orderIndex)})->second,
                                                            playerToWaypointIndex, state, navFile,
                                                            playerToEntryIndex, playerToExecuteStatus, orderIndex, teams[teamIndex], result);
            }
        }

        return result;
    }
};

#endif //CSKNOW_STRATEGY_DATA_H
