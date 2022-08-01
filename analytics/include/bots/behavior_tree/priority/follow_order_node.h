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

    namespace spacing {
        class NoMovementNode : public Node {
        public:
            NoMovementNode(Blackboard & blackboard) : Node(blackboard, "NoMovementNode") { };
            virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
        };

        class BaitConditionNode : public ConditionDecorator {
        public:
            BaitConditionNode(Blackboard & blackboard) :
                    ConditionDecorator(blackboard,make_unique<SequenceNode>(blackboard,
                                                                            Node::makeList(
                                                                                    make_unique<NoMovementNode>(blackboard)//,
                                                                                    //make_unique<movement::WaitNode>(blackboard, 0.5)
                                                                            ) ,"BaitNodesSequence"),
                                       "BaitConditionNode") { };
            virtual bool valid(const ServerState & state, TreeThinker &treeThinker) override;
        };

        class LurkConditionNode : public ConditionDecorator {
        public:
            LurkConditionNode(Blackboard & blackboard) :
                    ConditionDecorator(blackboard,make_unique<SequenceNode>(blackboard,
                                                                            Node::makeList(
                                                                                    make_unique<NoMovementNode>(blackboard)//,
                                                                                    //make_unique<movement::WaitNode>(blackboard, 0.5)
                                                                            ) ,"LurkNodesSequence"),
                                       "LurkConditionNode") { };
            virtual bool valid(const ServerState & state, TreeThinker &treeThinker) override;
        };

        class PushConditionNode : public ConditionDecorator {
        public:
            PushConditionNode(Blackboard & blackboard) :
                    ConditionDecorator(blackboard,make_unique<SequenceNode>(blackboard,
                                                                            Node::makeList(
                                                                                    make_unique<NoMovementNode>(blackboard)//,
                                                                                    //make_unique<movement::WaitNode>(blackboard, 0.5)
                                                                            ) ,"PushNodesSequence"),
                                       "PushConditionNode") { };
            virtual bool valid(const ServerState & state, TreeThinker &treeThinker) override;
        };
    }
}

class FollowOrderNode : public SequenceNode {
public:
    FollowOrderNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                                                        make_unique<follow::ComputeNavAreaNode>(blackboard),
                                                        make_unique<movement::PathingNode>(blackboard),
                                                        make_unique<follow::spacing::BaitConditionNode>(blackboard),
                                                        make_unique<follow::spacing::LurkConditionNode>(blackboard),
                                                        make_unique<follow::spacing::PushConditionNode>(blackboard)),
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
