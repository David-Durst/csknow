//
// Created by durst on 5/1/22.
//

#ifndef CSKNOW_ORDER_NODE_H
#define CSKNOW_ORDER_NODE_H

#include "bots/behavior_tree/node.h"
#include <map>
#include <memory>


namespace order {
    static vector<string> longToAPathPlaces = { "LongDoors", "LongA", "ARamp", "BombsiteA" };
    static vector<string> spawnToAPathPlaces = { "CTSpawn", "UnderA", "ARamp", "BombsiteA" };
    static vector<string> catToAPathPlace = { "Catwalk", "ShortStairs", "ExtendedA", "BombsiteA" };
    static vector<string> bDoorsToBPathPlaces = { "BDoors", "BombsiteB" };
    static vector<string> lowerTunsToBPathPlaces = { "LowerTunnel", "UpperTunnel", "BombsiteB" };
    static vector<string> outsideTunsToBPathPlaces = { "OutsideTunnel", "UpperTunnel", "BombsiteB" };

    class D2OrderNode : public Node {
        int32_t planRoundNumber = -1;
        int32_t playersAliveLastPlan = -1;
    public:
        D2OrderNode(Blackboard & blackboard) : Node(blackboard, "D2TaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class GeneralOrderNode : public Node {
    public:
        GeneralOrderNode(Blackboard & blackboard) : Node(blackboard, "GeneralTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class OrderNode : public SelectorNode {
public:
    OrderNode(Blackboard & blackboard) :
            SelectorNode(blackboard, Node::makeList(
                                                            make_unique<order::D2OrderNode>(blackboard)
                    ), "OrderNode") { };
};

#endif //CSKNOW_ORDER_NODE_H
