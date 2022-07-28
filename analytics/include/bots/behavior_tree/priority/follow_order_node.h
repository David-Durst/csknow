//
// Created by durst on 5/3/22.
//

#ifndef CSKNOW_FOLLOW_ORDER_NODE_H
#define CSKNOW_FOLLOW_ORDER_NODE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/pathing_node.h"
#include <map>
#include "bots/behavior_tree/condition_helper_node.h"

#define HOLD_DISTANCE 300.f
#define BAIT_DISTANCE 50.f

namespace follow {
    namespace compute_nav_area {
        class ComputeEntryNavAreaNode : public Node {
        public:
            ComputeEntryNavAreaNode(Blackboard & blackboard) : Node(blackboard, "ComputeEntryNavAreaNode") { };
            virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
        };

        class ComputeHoldNavAreaNode : public Node {
        public:
            ComputeHoldNavAreaNode(Blackboard & blackboard) : Node(blackboard, "ComputeHoldNavAreaNode") { };
            virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
        };
    }

    class ComputeNavAreaNode : public SelectorNode {
    public:
        ComputeNavAreaNode(Blackboard & blackboard) :
                SelectorNode(blackboard, Node::makeList(
                                     make_unique<TeamConditionDecorator>(
                                             blackboard, make_unique<compute_nav_area::ComputeEntryNavAreaNode>(blackboard),
                                             ENGINE_TEAM_CT),
                                     make_unique<compute_nav_area::ComputeHoldNavAreaNode>(blackboard)),
                             "ComputeNavAreaNode") { };
    };

    class BaitMovementNode : public Node {
    public:
        BaitMovementNode(Blackboard & blackboard) : Node(blackboard, "BaitMovementNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    class BaitConditionNode : public ConditionDecorator {
    public:
        BaitConditionNode(Blackboard & blackboard) : ConditionDecorator(blackboard,make_unique<SequenceNode>(blackboard,
                                                                                               Node::makeList(
                                                                                                       make_unique<BaitMovementNode>(blackboard),
                                                                                                       make_unique<movement::WaitNode>(blackboard, 0.5)
                                                                                               ) ,"BaitNodesSequence"),
                                                                         "BaitConditionNode") { };
        virtual bool valid(const ServerState & state, TreeThinker &treeThinker) override;
    };
}

class FollowOrderNode : public SequenceNode {
public:
    FollowOrderNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                                                        make_unique<follow::ComputeNavAreaNode>(blackboard),
                                                        make_unique<movement::PathingNode>(blackboard),
                                                        make_unique<follow::BaitConditionNode>(blackboard)),
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
