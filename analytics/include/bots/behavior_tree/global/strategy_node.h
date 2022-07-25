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
            {WaypointType::C4, "BombsiteB"}
    };
    static Waypoints offenseSpawnToAWaypoints = {
            {WaypointType::NavPlace, "CTSpawn"},
            {WaypointType::NavPlace, "UnderA"},
            {WaypointType::NavPlace, "ARamp"},
            {WaypointType::NavPlace, "BombsiteA"},
            {WaypointType::C4, "BombsiteB"}
    };
    static Waypoints offenseCatToAWaypoints = {
            {WaypointType::NavPlace, "Catwalk"},
            {WaypointType::NavPlace, "ShortStairs"},
            {WaypointType::NavPlace, "ExtendedA"},
            {WaypointType::NavPlace, "BombsiteA"},
            {WaypointType::C4, "BombsiteB"}
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
            {WaypointType::NavPlace, "BombsiteA"},
            {WaypointType::NavPlace, "ARamp"},
            {WaypointType::NavPlace, "LongA"},
            {WaypointType::NavPlace, "LongDoors"}
    };
    static Waypoints defenseSpawnToAWaypoints = {
            {WaypointType::NavPlace, "BombsiteA"},
            {WaypointType::NavPlace, "ARamp"},
            {WaypointType::NavPlace, "UnderA"},
            {WaypointType::NavPlace, "CTSpawn"}
    };
    static Waypoints defenseCatToAWaypoints = {
            {WaypointType::NavPlace, "BombsiteA"},
            {WaypointType::NavPlace, "ExtendedA"},
            {WaypointType::NavPlace, "ShortStairs"},
            {WaypointType::NavPlace, "Catwalk"}
    };
    static Waypoints defenseBDoorsToBWaypoints = {
            {WaypointType::NavPlace, "BombsiteB"},
            {WaypointType::NavAreas, "", "CustomBDoors", {9046, 9045, 9029, 9038}},
            {WaypointType::NavPlace, "MidDoors"}
    };
    static Waypoints defenseHoleToBWaypoints = {
            {WaypointType::NavPlace, "BombsiteB"},
            {WaypointType::NavPlace, "Hole"},
            {WaypointType::NavPlace, "MidDoors"}
    };
    static Waypoints defenseLowerTunsToBWaypoints = {
            {WaypointType::NavPlace, "BombsiteB"},
            {WaypointType::NavPlace, "UpperTunnel"},
            {WaypointType::NavPlace, "LowerTunnel"}
    };
    static Waypoints defenseSpawnToBWaypoints = {
            {WaypointType::NavPlace, "BombsiteB"},
            {WaypointType::NavPlace, "UpperTunnel"},
            {WaypointType::NavPlace, "OutsideTunnel"}
    };
    static vector<Order> aDefenseOrders{{defenseLongToAWaypoints}, {defenseSpawnToAWaypoints},
                                        {defenseCatToAWaypoints}};
    static vector<Order> bDefenseOrders{{defenseBDoorsToBWaypoints}, {defenseHoleToBWaypoints},
                                        {defenseLowerTunsToBWaypoints}, {defenseSpawnToBWaypoints}};

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
