//
// Created by durst on 8/22/22.
//

#ifndef CSKNOW_TEST_HEAD_POSITION_H
#define CSKNOW_TEST_HEAD_POSITION_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"

class HeadTrackingScript : public Script {
public:
    HeadTrackingScript(const ServerState & state) :
            Script("HeadTrackingScript", {{0, ENGINE_TEAM_T}},
                   {ObserveType::Absolute, 0, {1417.528564, 1412.913574, 54.766550}, {0., 90.}}) { }

    virtual void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                        make_unique<InitTestingRound>(blackboard, name),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0),
                                                                        make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id}, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({0., 0.})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 2.0)),
                                                                "HeadTrackingScript");
            Node::Ptr disableDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id})
            ));
            Node::Ptr MoveHead = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({0., -88.})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<Teleport>(blackboard, neededBots[0].id, state),
                    make_unique<movement::WaitNode>(blackboard, 5.0),
                    make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({0., 0.})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<Teleport>(blackboard, neededBots[0].id, state),
                    make_unique<movement::WaitNode>(blackboard, 5.0),
                    make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({0., 88.})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<Teleport>(blackboard, neededBots[0].id, state),
                    make_unique<movement::WaitNode>(blackboard, 5.0)
            ), "MoveHead");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id}),
                                                                                                make_unique<RepeatDecorator>(blackboard, std::move(MoveHead), false),
                                                                                                make_unique<movement::WaitNode>(blackboard, 600, false)),
                                                                                        "HeadTrackingCondition")),
                                                 "HeadTrackingSequence");
        }
    }
};

#endif //CSKNOW_TEST_HEAD_POSITION_H
