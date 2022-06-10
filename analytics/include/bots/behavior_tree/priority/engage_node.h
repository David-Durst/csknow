//
// Created by durst on 6/9/22.
//

#ifndef CSKNOW_ENGAGE_NODE_H
#define CSKNOW_ENGAGE_NODE_H
#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/movement_node.h"
#include <map>

namespace engage {
    class SelectTargetNode : public Node {
    public:
        SelectTargetNode(Blackboard & blackboard) : Node(blackboard, "SelectTargetNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class FollowOrderNode : public SequenceNode {
public:
    FollowOrderNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                                 make_unique<engage::SelectTargetNode>(blackboard),
                                 make_unique<movement::WaitNode>(blackboard, 0.5)),
                         "FollowOrderSelectorNode") { };
};

class EnemyEngageCheckNode : public ConditionDecorator {
public:
    EnemyEngageCheckNode(Blackboard & blackboard) : ConditionDecorator(blackboard,
                                                                        make_unique<FollowOrderNode>(blackboard),
                                                                        "EnemyEngageCheckNode") { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
};

#endif //CSKNOW_ENGAGE_NODE_H
