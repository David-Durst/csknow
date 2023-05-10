//
// Created by durst on 5/10/23.
//

#ifndef CSKNOW_TEST_BASIC_H
#define CSKNOW_TEST_BASIC_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"

class NavInsideNodeScript : public Script {
public:
    explicit NavInsideNodeScript(const ServerState &) :
            Script("NavInsideNodeScript", {{0, ENGINE_TEAM_T}}, {ObserveType::FirstPerson, 0}) { }

    void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Node::Ptr stillAndAtDestination = make_unique<RepeatDecorator>(blackboard,
                                                                           make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                                   make_unique<PosConstraint>(blackboard, neededBots[0].id,
                                                                                                              PosConstraintDimension::X, PosConstraintOp::LT,
                                                                                                              1070.),
                                                                                   make_unique<PosConstraint>(blackboard, neededBots[0].id,
                                                                                                              PosConstraintDimension::Y, PosConstraintOp::GT,
                                                                                                              680.),
                                                                                   make_unique<StandingStill>(blackboard, vector{neededBots[0].id}))),
                                                                            true);
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 0.3),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id},state),
                                                         make_unique<SetPos>(blackboard, Vec3({1062., 237., 85.}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ForceTargetPos>(blackboard, neededBots[0].id, Vec3({1002., 723., 7.})),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                std::move(stillAndAtDestination),
                                                                                                make_unique<movement::WaitNode>(blackboard, 20, false)),
                                                                                        "NavInsideNodeCondition")),
                                                 "NavInsideNodeSequence");
        }
    }
};

#endif //CSKNOW_TEST_BASIC_H
