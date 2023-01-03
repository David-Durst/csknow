//
// Created by steam on 7/1/22.
//

#ifndef CSKNOW_BASIC_AIM_H
#define CSKNOW_BASIC_AIM_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"

class KilledAfterTime : public Node {
    CSGOId sourceId, targetId;
    int32_t startFrame = INVALID_ID;
    double minTime;
    bool testScoped;

public:
    KilledAfterTime(Blackboard & blackboard, CSGOId sourceId, CSGOId targetId, double minTime, bool testScoped = false) :
            Node(blackboard, "ValidConditionNode"), sourceId(sourceId), targetId(targetId),
            minTime(minTime), testScoped(testScoped) { };

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & sourceClient = state.getClient(sourceId);
        const ServerState::Client & targetClient = state.getClient(targetId);
        if (playerNodeState[treeThinker.csgoId] == NodeState::Uninitialized) {
            startFrame = targetClient.lastFrame;
        }

        bool scopeCheck = !testScoped || sourceClient.isScoped;
        double timeSinceStart = (targetClient.lastFrame - startFrame) * state.tickInterval;
        if (!sourceClient.isAlive || !targetClient.isAlive) {
            playerNodeState[treeThinker.csgoId] = sourceClient.isAlive && !targetClient.isAlive &&
                    timeSinceStart > minTime && scopeCheck ? NodeState::Success : NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Running;
        return playerNodeState[treeThinker.csgoId];
    }

    void restart(const TreeThinker & treeThinker) override {
        Node::restart(treeThinker);
    }
};

struct PrintAim : public Node {
    int numTicks;
    PrintAim(Blackboard & blackboard, int numTicks) : Node(blackboard, "PrintAim"), numTicks(numTicks) { }

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.streamingManager.streamingEngagementAim.printAimTicks = numTicks;
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return NodeState::Success;
    }
};

struct ClearAimTargets : public Node {
    ClearAimTargets(Blackboard & blackboard) : Node(blackboard, "PrintAim") { }

    NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.streamingManager.streamingEngagementAim.currentClientTargetMap.clear();
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return NodeState::Success;
    }
};

class AimAndKillWithinTimeCheck : public Script {
public:
    explicit AimAndKillWithinTimeCheck(const ServerState &) :
            Script("AimAndKillWithinTimeCheck", {{0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

    void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id}, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1440.059204, 1112.913574, -8.766550}), Vec2({-178.035400, -1.805965})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -9.775691}), Vec2({-89.683349, 0.746031})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<RemoveGuns>(blackboard, neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<GiveItem>(blackboard, neededBots[0].id, state, "weapon_ak47"),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetCurrentItem>(blackboard, neededBots[0].id, state, "weapon_ak47"),
                                                         make_unique<ClearAimTargets>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 5),
                                                         make_unique<PrintAim>(blackboard, 10)),
                                                 "AimAndKillWithinTimeCheckSetup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                        std::move(setupCommands),
                                                        make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id})
                    ), "AimAndKillDisableDuringSetup");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<KilledAfterTime>(blackboard, neededBots[0].id, neededBots[1].id, 0.5),
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[1].id}),
                                                                                                make_unique<movement::WaitNode>(blackboard, 4, false)),
                                                                                        "AimAndKillCondition")),
                                                 "AimAndKillSequence");
        }
    }
};

// scoping takes time (need to wait for scope animation to happen, need to predict aim), not gonna deal with it now
class ScopedAimAndKillWithinTimeCheck : public Script {
public:
    explicit ScopedAimAndKillWithinTimeCheck(const ServerState &) :
            Script("ScopedAimAndKillWithinTimeCheck", {{0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

    void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                        make_unique<InitTestingRound>(blackboard, name),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0),
                                                                        make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id}, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1440.059204, 1112.913574, -8.766550}), Vec2({-178.035400, -1.805965})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -9.775691}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<RemoveGuns>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<GiveItem>(blackboard, neededBots[0].id, state, "weapon_awp"),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetCurrentItem>(blackboard, neededBots[0].id, state, "weapon_awp"),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0)),
                                                                "ScopedAimAndKillWithinTimeCheckSetup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id})
            ), "ScopedAimAndKillDisableDuringSetup");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<KilledAfterTime>(blackboard, neededBots[0].id, neededBots[1].id, 0.5, true),
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[1].id}),
                                                                                                make_unique<movement::WaitNode>(blackboard, 3, false)),
                                                                                        "ScopedAimAndKillCondition")),
                                                 "ScopedAimAndKillSequence");
        }
    }
};

#endif //CSKNOW_BASIC_AIM_H
