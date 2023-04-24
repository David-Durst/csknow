//
// Created by durst on 5/3/22.
//

#ifndef CSKNOW_FOLLOW_ORDER_NODE_H
#define CSKNOW_FOLLOW_ORDER_NODE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/pathing_node.h"
#include <map>
#include "bots/behavior_tree/condition_helper_node.h"
#include "bots/behavior_tree/priority/model_nav_data.h"

#define HOLD_DISTANCE 300.f

namespace follow {
    namespace compute_nav_area {
        class ComputeModelNavAreaNode : public Node {
        public:
            ComputeModelNavAreaNode(Blackboard & blackboard) : Node(blackboard, "ComputeModelNavAreaNode") { };
            PlaceIndex computePlaceProbabilistic(const ServerState & state, const Order & curOrder,
                                                 AreaId curAreaId, CSGOId csgoId, ModelNavData & modelNavData);
            void computeAreaProbabilistic(const ServerState & state, Priority & curPriority, PlaceIndex nextPlace,
                                          CSGOId csgoId, ModelNavData & modelNavData);
            virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
        };

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
                                     make_unique<compute_nav_area::ComputeModelNavAreaNode>(blackboard),
                                     make_unique<TeamConditionDecorator>(
                                             blackboard, make_unique<compute_nav_area::ComputeEntryNavAreaNode>(blackboard),
                                             ENGINE_TEAM_CT),
                                     make_unique<compute_nav_area::ComputeHoldNavAreaNode>(blackboard)),
                             "ComputeNavAreaNode") { };
    };

    class ComputeNonDangerAimAreaNode : public Node {
    public:
        ComputeNonDangerAimAreaNode(Blackboard & blackboard) : Node(blackboard, "ComputeNonDangerAimAreaNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    namespace spacing {
        class LearnedSpacingNode : public Node {
        public:
            LearnedSpacingNode(Blackboard & blackboard) : Node(blackboard, "LearnedSpacingNode") { };
            virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
        };

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

    class SpacingNode : public SelectorNode {
    public:
        SpacingNode(Blackboard & blackboard) :
            SelectorNode(blackboard, Node::makeList(
                make_unique<spacing::LearnedSpacingNode>(blackboard),
                make_unique<spacing::BaitConditionNode>(blackboard),
                make_unique<spacing::LurkConditionNode>(blackboard),
                make_unique<spacing::PushConditionNode>(blackboard)),
        "SpacingSelectorNode") { };
    };

    class EnemiesAliveTaskNode : public Node {
    public:
        EnemiesAliveTaskNode(Blackboard & blackboard) : Node(blackboard, "EnemiesAliveTaskNode") { };
        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };
}


class FollowOrderNode : public SequenceNode {
public:
    FollowOrderNode(Blackboard & blackboard) :
            SequenceNode(blackboard, Node::makeList(
                                                        make_unique<follow::ComputeNavAreaNode>(blackboard),
                                                        make_unique<follow::ComputeNonDangerAimAreaNode>(blackboard),
                                                        make_unique<movement::PathingNode>(blackboard),
                                                        make_unique<follow::EnemiesAliveTaskNode>(blackboard),
                                                        make_unique<follow::SpacingNode>(blackboard)),
                                        "FollowOrderSequenceNode") { };
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
