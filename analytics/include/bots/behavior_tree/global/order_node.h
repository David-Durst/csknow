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
    static vector<string> bDoorsToBPathPlaces = { "MidDoors", "BDoors", "BombsiteB" };
    static vector<string> lowerTunsToBPathPlaces = { "LowerTunnel", "UpperTunnel", "BombsiteB" };
    static vector<string> outsideTunsToBPathPlaces = { "OutsideTunnel", "UpperTunnel", "BombsiteB" };

    class OrderNode : public Node {
        int32_t planRoundNumber = -1;
        int32_t playersAliveLastPlan = -1;
    public:
        OrderNode(Blackboard & blackboard) : Node(blackboard, "OrderNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

#endif //CSKNOW_ORDER_NODE_H
