//
// Created by steam on 7/5/22.
//

#ifndef CSKNOW_COMMMUNICATION_H
#define CSKNOW_COMMMUNICATION_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"


class CommunicationAimCheck : public Script {
public:
    explicit CommunicationAimCheck(const ServerState &) :
            Script("CommunicationAimCheck", {{0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

    void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                        make_unique<InitTestingRound>(blackboard, name),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0),
                                                                        make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.3),
                                                                        make_unique<SetPos>(blackboard, Vec3({1031.611084, 962.737915, -0.588848}), Vec2({90., 0.})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1440.059204, 1112.913574, -8.766550}), Vec2({90., 0.})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -9.775691}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<movement::WaitNode>(blackboard, 2.0)),
                                                                "CommunicationAimSetup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, false)
            ), "CommunicationAimDisableDuringSetup");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<AimingAt>(blackboard, neededBots[0].id, neededBots[2].id),
                                                                                                make_unique<Firing>(blackboard, neededBots[0].id, true),
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, false),
                                                                                                make_unique<movement::WaitNode>(blackboard, 0.8)),
                                                                                        //make_unique<movement::WaitNode>(blackboard, 0.8)),
                                                                                        "CommunicationAimCondition")),
                                                 "CommunicationAimSequence");
        }
    }
};

class CommunicationForgetCheck : public Script {
public:
    explicit CommunicationForgetCheck(const ServerState &) :
            Script("CommunicationForgetCheck", {{0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

    void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                        make_unique<InitTestingRound>(blackboard, name),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0),
                                                                        make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.3),
                                                                        make_unique<SetPos>(blackboard, Vec3({1031.611084, 962.737915, -0.588848}), Vec2({90., 0.})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1440.059204, 1112.913574, -8.766550}), Vec2({90., 0.})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -9.775691}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0),
                                                                        make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[2].id}, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 2.0)),
                                                                "CommunicationForgetSetup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, false)
            ), "CommunicationAimDisableDuringSetup");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<AimingAt>(blackboard, neededBots[0].id, neededBots[2].id, true),
                                                                                                make_unique<Firing>(blackboard, neededBots[0].id, true),
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, false),
                                                                                                make_unique<movement::WaitNode>(blackboard, 0.8)),
                                                                 //make_unique<movement::WaitNode>(blackboard, 0.8)),
                                                                                        "CommunicationAimCondition")),
                                                 "CommunicationAimSequence");
        }
    }
};

class CommunicationIgnoreCheck : public Script {
public:
    explicit CommunicationIgnoreCheck(const ServerState &) :
            Script("CommunicationIgnoreCheck", {{0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

    void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                        make_unique<InitTestingRound>(blackboard, name),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0),
                                                                        make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.3),
                                                                        make_unique<SetPos>(blackboard, Vec3({-420.947021, 1067.724365, -69.326324}), Vec2({-78.701942, -7.463999})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1440.059204, 1112.913574, -8.766550}), Vec2({90., 0.})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -9.775691}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<movement::WaitNode>(blackboard, 2.0)),
                                                                "CommunicationIgnoreSetup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, false)
            ), "CommunicationIgnoreDisableDuringSetup");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<AimingAt>(blackboard, neededBots[0].id, neededBots[2].id, true),
                                                                                                make_unique<Firing>(blackboard, neededBots[0].id, true),
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[1].id, neededBots[2].id}),
                                                                                                make_unique<movement::WaitNode>(blackboard, 0.8)),
                                                                                        "CommunicationIgnoreCondition")),
                                                 "CommunicationIgnoreSequence");
        }
    }
};

#endif //CSKNOW_COMMMUNICATION_H
