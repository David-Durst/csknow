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
    static Waypoints offenseLongToAWaypoints = {
            {WaypointType::NavPlace, "LongDoors"},
            {WaypointType::NavPlace, "LongA"},
            {WaypointType::NavPlace, "ARamp"},
            {WaypointType::NavPlace, "BombsiteA"},
            {WaypointType::C4, "BombsiteA"}
    };
    static Waypoints offenseSpawnToAWaypoints = {
            {WaypointType::NavPlace, "CTSpawn"},
            {WaypointType::NavPlace, "UnderA"},
            {WaypointType::NavPlace, "ARamp"},
            {WaypointType::NavPlace, "BombsiteA"},
            {WaypointType::C4, "BombsiteA"}
    };
    static Waypoints offenseCatToAWaypoints = {
            {WaypointType::NavPlace, "Catwalk"},
            {WaypointType::NavPlace, "ShortStairs"},
            {WaypointType::NavPlace, "ExtendedA"},
            {WaypointType::NavPlace, "BombsiteA"},
            {WaypointType::C4, "BombsiteA"}
    };
    static Waypoints offenseBDoorsToBWaypoints = {
            {WaypointType::NavPlace, "MidDoors"},
            {WaypointType::NavAreas, "", "CustomBDoors", {9046, 9045, 9029, 9038}},
            {WaypointType::NavPlace, "BombsiteB"},
            {WaypointType::C4, "BombsiteB"}
    };
    static Waypoints offenseHoleToBWaypoints = {
            {WaypointType::NavPlace, "MidDoors"},
            {WaypointType::NavPlace, "Hole"},
            {WaypointType::NavPlace, "BombsiteB"},
            {WaypointType::C4, "BombsiteB"}
    };
    static Waypoints offenseLowerTunsToBWaypoints = {
            {WaypointType::NavPlace, "LowerTunnel"},
            {WaypointType::NavPlace, "UpperTunnel"},
            {WaypointType::NavPlace, "BombsiteB"},
            {WaypointType::C4, "BombsiteB"}
    };
    static Waypoints offenseSpawnToBWaypoints = {
            {WaypointType::NavPlace, "OutsideTunnel"},
            {WaypointType::NavPlace, "UpperTunnel"},
            {WaypointType::NavPlace, "BombsiteB"},
            {WaypointType::C4, "BombsiteB"}
    };
    static vector<Order> aOffenseOrders{{offenseLongToAWaypoints}, {offenseSpawnToAWaypoints},
                                        {offenseCatToAWaypoints}};

    static vector<Order> bOffenseOrders{{offenseBDoorsToBWaypoints}, {offenseHoleToBWaypoints},
                                        {offenseLowerTunsToBWaypoints}, {offenseSpawnToBWaypoints}};

    // t sequences of waypoint labels
    static Waypoints defenseLongToAWaypoints = {
            {WaypointType::C4, "BombsiteA"},
            {WaypointType::HoldPlace, "ARamp", "", {}, false},
            {WaypointType::HoldAreas, "", "ACar", {1794, 1799}, false},
            {WaypointType::HoldAreas, "", "LongACorner", {4170}, true},
            {WaypointType::HoldPlace, "Pit", "", {5211, 8286, 8287}, true},
            {WaypointType::ChokeAreas, "", "BotALong", {4170}, false},
            {WaypointType::ChokeAreas, "", "LongDoorDumpster", {3653}, true}
    };
    static Waypoints defenseSpawnToAWaypoints = {
            {WaypointType::C4, "BombsiteA"},
            {WaypointType::HoldPlace, "BombsiteA", "", {}, false},
            {WaypointType::HoldPlace, "LongA", "", {}, true},
            {WaypointType::ChokePlace, "UnderA", "", {}, false},
            {WaypointType::ChokePlace, "CTSpawn", "", {}, true}
    };
    static Waypoints defenseCatToAWaypoints = {
            {WaypointType::C4, "BombsiteA"},
            {WaypointType::HoldPlace, "BombsiteA", "", {}, false},
            {WaypointType::HoldPlace, "ExtendedA", "", {}, true},
            {WaypointType::ChokeAreas, "", "StairsBox", {4048}, false},
            {WaypointType::ChokePlace, "ShortStairs", "", {}, true}
    };
    static Waypoints defenseBDoorsToBWaypoints = {
            {WaypointType::C4, "BombsiteB"},
            {WaypointType::HoldAreas, "", "BCar", {1642}, false},
            {WaypointType::HoldAreas, "", "BDoorsBox", {1911}, true},
            {WaypointType::ChokePlace, "BDoors", "", {}, false},
            {WaypointType::ChokePlace, "MidDoors", "", {}, true}
    };
    static Waypoints defenseHoleToBWaypoints = {
            {WaypointType::C4, "BombsiteB"},
            {WaypointType::HoldPlace, "BombsiteB", "", {}, false},
            {WaypointType::HoldPlace, "Hole", "", {}, true},
            {WaypointType::ChokePlace, "Hole", "", {}, false},
            {WaypointType::ChokePlace, "MidDoors", "", {}, true}
    };
    static Waypoints defenseTunsToBWaypoints = {
            {WaypointType::C4, "BombsiteB"},
            {WaypointType::HoldAreas, "", "BackPlat", {4010, 6813}, false},
            {WaypointType::HoldPlace, "UpperTunnel", "", {}, true},
            {WaypointType::ChokePlace, "UpperTunnel", "", {}, false},
            {WaypointType::ChokePlace, "TunnelStairs", "", {}, true}
    };
    static vector<Order> aDefenseOrders{{defenseLongToAWaypoints}, {defenseSpawnToAWaypoints},
                                        {defenseCatToAWaypoints}};
    static vector<Order> bDefenseOrders{{defenseBDoorsToBWaypoints}, {defenseHoleToBWaypoints},
                                        {defenseTunsToBWaypoints}};

    class CreateOrdersNode : public Node {
        int32_t planRoundNumber = -1;
        int32_t playersAliveLastPlan = -1;
    public:
        CreateOrdersNode(Blackboard & blackboard) : Node(blackboard, "CreateOrdersNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class AssignPlayersToOrders : public Node {
    public:
        AssignPlayersToOrders(Blackboard & blackboard) : Node(blackboard, "AssignPlayersToOrders") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class CreateStrategyNode : public SequenceNode {
public:
    CreateStrategyNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                    make_unique<strategy::CreateOrdersNode>(blackboard),
                    make_unique<strategy::AssignPlayersToOrders>(blackboard)
            ), "CreateStrategyNode") { };
};

#endif //CSKNOW_STRATEGY_NODE_H
