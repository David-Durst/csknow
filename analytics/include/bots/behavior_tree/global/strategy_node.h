//
// Created by durst on 5/1/22.
//

#ifndef CSKNOW_STRATEGY_NODE_H
#define CSKNOW_STRATEGY_NODE_H

#include "bots/behavior_tree/global/strategy_data.h"
#include "bots/behavior_tree/node.h"
#include <map>
#include <memory>

namespace strategy {
    // ct sequences of waypoint labels
    const Waypoints offenseLongToAWaypoints = {
        {WaypointType::NavPlace, "LongDoors"},
        {WaypointType::NavPlace, "LongA"},
        {WaypointType::NavPlace, "ARamp"},
        {WaypointType::NavPlace, "BombsiteA"},
        {WaypointType::C4,       "BombsiteA"}
    };
    const Waypoints offenseSpawnToAWaypoints = {
        {WaypointType::NavPlace, "CTSpawn"},
        {WaypointType::NavPlace, "UnderA"},
        {WaypointType::NavPlace, "ARamp"},
        {WaypointType::NavPlace, "BombsiteA"},
        {WaypointType::C4,       "BombsiteA"}
    };
    const Waypoints offenseCatToAWaypoints = {
        {WaypointType::NavPlace, "ShortStairs"},
        {WaypointType::NavPlace, "ExtendedA"},
        {WaypointType::NavPlace, "BombsiteA"},
        {WaypointType::C4,       "BombsiteA"}
    };
    const Waypoints offenseBDoorsToBWaypoints = {
        {WaypointType::NavAreas, "", "CustomBDoors", {9046, 9045, 9029, 9038}},
        {WaypointType::NavPlace, "BombsiteB"},
        {WaypointType::C4,       "BombsiteB"}
    };
    const Waypoints offenseHoleToBWaypoints = {
        {WaypointType::NavPlace, "Hole"},
        {WaypointType::NavPlace, "BombsiteB"},
        {WaypointType::C4,       "BombsiteB"}
    };
    const Waypoints offenseTunsToBWaypoints = {
        {WaypointType::NavAreas, "", "CustomUpperTunnel", {1229, 1230, 1226}},
        {WaypointType::NavPlace, "BombsiteB"},
        {WaypointType::C4,       "BombsiteB"}
    };
    const vector<Order> aOffenseOrders{{offenseLongToAWaypoints}, {offenseSpawnToAWaypoints},
                                        {offenseCatToAWaypoints}};

    const vector<Order> bOffenseOrders{{offenseBDoorsToBWaypoints}, {offenseHoleToBWaypoints},
                                        {offenseTunsToBWaypoints}};

    // t sequences of waypoint labels
    const Waypoints defenseLongToAWaypoints = {
        {WaypointType::C4,         "BombsiteA"},
        {WaypointType::HoldPlace,  "ARamp", "",                 {},                 false},
        {WaypointType::HoldAreas,  "",      "ACar",             {1794, 1799},       false},
        {WaypointType::HoldAreas,  "",      "LongACorner",      {4170},             true},
        {WaypointType::HoldPlace,  "Pit",   "",                 {5211, 8286, 8287}, true},
        {WaypointType::ChokeAreas, "",      "BotALong",         {4170},             false},
        {WaypointType::ChokeAreas, "",      "LongDoorDumpster", {3653},             true}
    };
    const Waypoints defenseSpawnToAWaypoints = {
        {WaypointType::C4,         "BombsiteA"},
        {WaypointType::HoldPlace,  "BombsiteA", "", {}, false},
        {WaypointType::HoldPlace,  "LongA",     "", {}, true},
        {WaypointType::ChokePlace, "UnderA",    "", {}, false},
        {WaypointType::ChokePlace, "CTSpawn",   "", {}, true}
    };
    const Waypoints defenseCatToAWaypoints = {
        {WaypointType::C4,         "BombsiteA"},
        {WaypointType::HoldPlace,  "BombsiteA",   "",          {},     false},
        {WaypointType::HoldPlace,  "ExtendedA",   "",          {},     true},
        {WaypointType::ChokeAreas, "",            "StairsBox", {4048}, false},
        {WaypointType::ChokePlace, "ShortStairs", "",          {},     true}
    };
    const Waypoints defenseBDoorsToBWaypoints = {
        {WaypointType::C4,         "BombsiteB"},
        {WaypointType::HoldAreas,  "",         "BCar",      {1642}, false},
        {WaypointType::HoldAreas,  "",         "BDoorsBox", {1911}, true},
        {WaypointType::ChokePlace, "BDoors",   "",          {},     false},
        {WaypointType::ChokePlace, "MidDoors", "",          {},     true}
    };
    const Waypoints defenseHoleToBWaypoints = {
        {WaypointType::C4,         "BombsiteB"},
        {WaypointType::HoldPlace,  "BombsiteB", "", {}, false},
        {WaypointType::HoldPlace,  "Hole",      "", {}, true},
        {WaypointType::ChokePlace, "Hole",      "", {}, false},
        {WaypointType::ChokePlace, "MidDoors",  "", {}, true}
    };
    const Waypoints defenseTunsToBWaypoints = {
        {WaypointType::C4,         "BombsiteB"},
        {WaypointType::HoldAreas,  "",             "BackPlat", {4010, 6813}, false},
        {WaypointType::HoldPlace,  "UpperTunnel",  "",         {},           true},
        {WaypointType::ChokePlace, "UpperTunnel",  "",         {},           false},
        {WaypointType::ChokePlace, "TunnelStairs", "",         {},           true}
    };
    const vector<Order> aDefenseOrders{{defenseLongToAWaypoints}, {defenseSpawnToAWaypoints},
                                        {defenseCatToAWaypoints}};
    const vector<Order> bDefenseOrders{{defenseBDoorsToBWaypoints}, {defenseHoleToBWaypoints},
                                        {defenseTunsToBWaypoints}};

    class CreateOrdersNode : public Node {
        RoundNumber planRoundNumber = -1;
        int32_t playersAliveLastPlan = -1;
        size_t ticksSinceLastOrder = 0;
        vector<CSGOId> ctPlayers = {}, tPlayers = {};
    public:
        explicit CreateOrdersNode(Blackboard & blackboard) : Node(blackboard, "CreateOrdersNode") { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class AssignPlayersToOrders : public Node {
    public:
        explicit AssignPlayersToOrders(Blackboard & blackboard) : Node(blackboard, "AssignPlayersToOrders") { };
        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
        bool assignPlayerToOrderProbabilistic(const ServerState::Client client, bool plantedA);
    };
}

class CreateStrategyNode : public SequenceNode {
public:
    explicit CreateStrategyNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                    make_unique<strategy::CreateOrdersNode>(blackboard),
                    make_unique<strategy::AssignPlayersToOrders>(blackboard)
            ), "CreateStrategyNode") { };
};

#endif //CSKNOW_STRATEGY_NODE_H
