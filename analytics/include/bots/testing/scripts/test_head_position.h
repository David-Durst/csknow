//
// Created by durst on 8/22/22.
//

#ifndef CSKNOW_TEST_HEAD_POSITION_H
#define CSKNOW_TEST_HEAD_POSITION_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"

class DrawHeadPos : public Command {
    CSGOId playerId;
    Vec3 visOffset;
    double radius;

public:
    DrawHeadPos(Blackboard & blackboard, CSGOId playerId, Vec3 visOffset = {0., -10., 0.}, double radius = 2.) :
            Command(blackboard, "DrawHeadPos"), playerId(playerId), visOffset(visOffset), radius(radius) { };

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_drawAABB 3.5 ";

        const ServerState::Client & client = state.getClient(playerId);
        Vec3 headCoordinates =
                getCenterHeadCoordinatesForPlayer(client.getEyePosForPlayer(), client.getCurrentViewAngles(), client.duckAmount);
        result << headCoordinates.x - radius + visOffset.x << " "
            << headCoordinates.y - radius + visOffset.y << " "
            << headCoordinates.z - radius + visOffset.z << " "
            << headCoordinates.x + radius + visOffset.x << " "
            << headCoordinates.y + radius + visOffset.y << " "
            << headCoordinates.z + radius + visOffset.z;

        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

Node::Ptr getMoveHeadNode(Blackboard & blackboard, ServerState & state, const vector<NeededBot> & neededBots) {
    return make_unique<SequenceNode>(blackboard, Node::makeList(
            make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({0., -88.})),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<Teleport>(blackboard, neededBots[0].id, state),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<DrawHeadPos>(blackboard, neededBots[0].id),
            make_unique<movement::WaitNode>(blackboard, 5.0),
            make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({0., 0.})),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<Teleport>(blackboard, neededBots[0].id, state),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<DrawHeadPos>(blackboard, neededBots[0].id),
            make_unique<movement::WaitNode>(blackboard, 5.0),
            make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({0., 88.})),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<Teleport>(blackboard, neededBots[0].id, state),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<DrawHeadPos>(blackboard, neededBots[0].id),
            make_unique<movement::WaitNode>(blackboard, 5.0),
            // angled
            make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({-45., -88.})),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<Teleport>(blackboard, neededBots[0].id, state),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<DrawHeadPos>(blackboard, neededBots[0].id),
            make_unique<movement::WaitNode>(blackboard, 5.0),
            make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({-45., 0.})),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<Teleport>(blackboard, neededBots[0].id, state),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<DrawHeadPos>(blackboard, neededBots[0].id),
            make_unique<movement::WaitNode>(blackboard, 5.0),
            make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({-45., 88.})),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<Teleport>(blackboard, neededBots[0].id, state),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<DrawHeadPos>(blackboard, neededBots[0].id),
            make_unique<movement::WaitNode>(blackboard, 5.0),
            // behind
            make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({180., -88.})),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<Teleport>(blackboard, neededBots[0].id, state),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<DrawHeadPos>(blackboard, neededBots[0].id),
            make_unique<movement::WaitNode>(blackboard, 5.0),
            make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({180., 0.})),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<Teleport>(blackboard, neededBots[0].id, state),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<DrawHeadPos>(blackboard, neededBots[0].id),
            make_unique<movement::WaitNode>(blackboard, 5.0),
            make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -7.28}), Vec2({180., 88.})),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<Teleport>(blackboard, neededBots[0].id, state),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<DrawHeadPos>(blackboard, neededBots[0].id),
            make_unique<movement::WaitNode>(blackboard, 5.0)
    ), "MoveHead");
}

class HeadTrackingScript : public Script {
public:
    explicit HeadTrackingScript(const ServerState &) :
            Script("HeadTrackingScript", {{0, ENGINE_TEAM_T}},
                   {ObserveType::Absolute, 0, {1417.528564, 1562.913574, 54.766550}, {0., 90.}}) { }

    void initialize(Tree & tree, ServerState & state) override {
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
            Node::Ptr moveHead = getMoveHeadNode(blackboard, state, neededBots);
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id}),
                                                                                                std::move(moveHead)
                                                         ), "HeadTrackingCondition")),
                                                 "HeadTrackingSequence");
        }
    }
};


class CrouchedHeadTrackingScript : public Script {
public:
    explicit CrouchedHeadTrackingScript(const ServerState &) :
            Script("CrouchedHeadTrackingScript", {{0, ENGINE_TEAM_T}},
                   {ObserveType::Absolute, 0, {1417.528564, 1562.913574, 44.766550}, {0., 90.}}) { }

    void initialize(Tree & tree, ServerState & state) override {
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
                                                                "CrouchedHeadTrackingScript");
            Node::Ptr disableDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id})
            ));
            Node::Ptr moveCrouchedHead = getMoveHeadNode(blackboard, state, neededBots);
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id}),
                                                                                                std::move(moveCrouchedHead),
                                                                                                make_unique<ForceActionsNode>(blackboard, vector{neededBots[0].id}, IN_DUCK)
                                                                                        ), "CrouchedHeadTrackingCondition")),
                                                 "CrouchedHeadTrackingSequence");
        }
    }
};
#endif //CSKNOW_TEST_HEAD_POSITION_H
