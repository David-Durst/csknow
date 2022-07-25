//
// Created by durst on 5/2/22.
//

#ifndef CSKNOW_STRATEGY_DATA_H
#define CSKNOW_STRATEGY_DATA_H
#include "bots/load_save_bot_data.h"
#include "navmesh/nav_file.h"
#include "bots/analysis/nav_file_helpers.h"
#include <sstream>
#include <algorithm>
using std::stringstream;
using std::map;

enum class WaypointType {
    NavPlace,
    NavAreas, // sometimes want to force a collection of areas without a name
    Player,
    C4,
    NUM_WAYPOINTS
};

struct Waypoint {
    WaypointType type;

    // use placeName if type of NavPlace, playerId if type of is player, placeName for site if c4
    // not using union because that prevented automatic destructor definition, and this is trivial amount of extra data
    string placeName;
    string customAreasName;
    set<AreaId> areaIds;
    CSGOId playerId;
};

typedef vector<Waypoint> Waypoints;
/**
 * Order specifies how to to navigate the map as part of a strategy
 */
struct Order {
    Waypoints waypoints;
    // what about chains of operations (like switching once plant happens)?

    void print(const vector<CSGOId> followers, const map<CSGOId, int64_t> & playerToWaypointIndex,
               const ServerState & state, const nav_mesh::nav_file & navFile,
               const map<CSGOId, int32_t> & playerToEntryIndex, size_t orderIndex, vector<string> & result) const {
        stringstream waypointsStream;
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
                case WaypointType::Player:
                    typeString = "Player";
                    dataString = state.getPlayerString(waypoint.playerId);
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
        followersStream << orderIndex << " followers: <follower, waypoint index, entry index> : [";
        for (const auto & follower : followers) {
            followersStream << "<" << state.getPlayerString(follower) << ", "
                << playerToWaypointIndex.find(follower)->second << ", "
                << playerToEntryIndex.find(follower)->second << ">; ";
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
            throw std::runtime_error( "getPlayersForOrder bad order id" );
        }
    }

    const Order & getOrder(OrderId orderId) const {
        if (orderId.team == ENGINE_TEAM_T) {
            return tOrders[orderId.index];
        }
        else if (orderId.team == ENGINE_TEAM_CT) {
            return ctOrders[orderId.index];
        }
        throw std::runtime_error( "getOrderForPlayer bad player id" );
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

    OrderId addOrder(TeamId team, Order order) {
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

    int64_t maxTeamWaypoint(const ServerState & state, TeamId team) {
        int64_t result = INVALID_ID;
        for (const auto & csgoId : state.getPlayersOnTeam(team)) {
            if (state.getClient(csgoId).team == team && playerToWaypointIndex[csgoId] > result) {
                result = playerToWaypointIndex[csgoId];
            }
        }
        return result;
    }

    vector<string> print(const ServerState & state, const nav_mesh::nav_file & navFile) const {
        vector<string> result;

        vector<TeamId> teams{ENGINE_TEAM_T, ENGINE_TEAM_CT};
        vector<vector<Order>> bothTeamOrders{tOrders, ctOrders};

        for (size_t teamIndex = 0; teamIndex < teams.size(); teamIndex++) {
            for (size_t orderIndex = 0; orderIndex < bothTeamOrders[teamIndex].size(); orderIndex++) {
                bothTeamOrders[teamIndex][orderIndex].print(orderToPlayers.find({teams[teamIndex], static_cast<int64_t>(orderIndex)})->second,
                                                            playerToWaypointIndex, state, navFile,
                                                            playerToEntryIndex, orderIndex, result);
            }
        }

        return result;
    }
};

#endif //CSKNOW_STRATEGY_DATA_H
