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
    int32_t startFrame;
    double minTime;

public:
    KilledAfterTime(Blackboard & blackboard, CSGOId sourceId, CSGOId targetId, double minTime) :
            Node(blackboard, "ValidConditionNode"), sourceId(sourceId), targetId(targetId), minTime(minTime) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & sourceClient = state.getClient(sourceId);
        const ServerState::Client & targetClient = state.getClient(targetId);
        if (playerNodeState[treeThinker.csgoId] == NodeState::Uninitialized) {
            startFrame = targetClient.lastFrame;
        }

        double timeSinceStart = (targetClient.lastFrame - startFrame) * state.tickInterval;
        if (!sourceClient.isAlive || !targetClient.isAlive) {
            playerNodeState[treeThinker.csgoId] = sourceClient.isAlive && !targetClient.isAlive && timeSinceStart > minTime ? NodeState::Success : NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Running;
        return playerNodeState[treeThinker.csgoId];
    }

    virtual void restart(const TreeThinker & treeThinker) override {
        Node::restart(treeThinker);
    }
};

class AimAndKillWithinTimeCheck : public Script {
public:
    AimAndKillWithinTimeCheck(const ServerState & state) :
            Script("AimAndKillWithinTimeCheck", {{0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

    virtual void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard)  {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            vector<string> aToCatPathPlace(order::catToAPathPlace.rbegin(), order::catToAPathPlace.rend());
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard),
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
                                                         make_unique<movement::WaitNode>(blackboard, 0.1)),
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
                                                                                                make_unique<movement::WaitNode>(blackboard, 300, false)),
                                                                                        "AimAndKillCondition")),
                                                 "AimAndKillSequence");
        }
    }
};

#endif //CSKNOW_BASIC_AIM_H
