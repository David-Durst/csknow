//
// Created by durst on 5/3/22.
//

#ifndef CSKNOW_FOLLOW_ORDER_NODE_H
#define CSKNOW_FOLLOW_ORDER_NODE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/pathing_node.h"
#include <map>

#define STUCK_TICKS_THRESHOLD 10
#define BAIT_DISTANCE 50.f

namespace follow {
    class ComputeObjectiveAreaNode : public Node {
    public:
        ComputeObjectiveAreaNode(Blackboard & blackboard) : Node(blackboard, "ComputeObstaclesTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class BaitWaitNode : public ConditionDecorator {
    public:
        BaitWaitNode(Blackboard & blackboard) : ConditionDecorator(blackboard,
                                                                         make_unique<movement::WaitNode>(blackboard, 0.5),
                                                                         "BaitWaitDecorator") { };
        virtual bool valid(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class FollowOrderNode : public SequenceNode {
public:
    FollowOrderNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                                                        make_unique<follow::ComputeObjectiveAreaNode>(blackboard),
                                                        make_unique<movement::PathingNode>(blackboard),
                                                        make_unique<follow::BaitWaitNode>(blackboard)),
                                        "FollowOrderSelectorNode") { };
};

class NoEnemyOrderCheckNode : public ConditionDecorator {
public:
    NoEnemyOrderCheckNode(Blackboard & blackboard) : ConditionDecorator(blackboard,
                                                               make_unique<FollowOrderNode>(blackboard),
                                                               "NoEnemyOrderCheckNode") { };
    virtual bool valid(const ServerState & state, TreeThinker & treeThinker) override {
        return state.getVisibleEnemies(treeThinker.csgoId).empty();
    }
};

#endif //CSKNOW_FOLLOW_ORDER_NODE_H
