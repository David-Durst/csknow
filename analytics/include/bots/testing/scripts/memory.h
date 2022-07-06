//
// Created by steam on 7/5/22.
//

#ifndef CSKNOW_MEMORY_H
#define CSKNOW_MEMORY_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"

class AimingAt : public Node {
    CSGOId sourceId, targetId;

public:
    AimingAt(Blackboard & blackboard, CSGOId sourceId, CSGOId targetId) :
            Node(blackboard, "AimingAtNode"), sourceId(sourceId), targetId(targetId) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & sourceClient = state.getClient(sourceId);
        const ServerState::Client & targetClient = state.getClient(targetId);

        // check if aiming at enemy anywhere
        Ray eyeCoordinates = getEyeCoordinatesForPlayerGivenEyeHeight(
                sourceClient.getEyePosForPlayer(),
                sourceClient.getCurrentViewAnglesWithAimpunch());

        AABB targetAABB = getAABBForPlayer(targetClient.getFootPosForPlayer());
        double hitt0, hitt1;
        bool aimingAtEnemy = intersectP(targetAABB, eyeCoordinates, hitt0, hitt1);

        if (aimingAtEnemy) {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        return playerNodeState[treeThinker.csgoId];
    }

    virtual void restart(const TreeThinker & treeThinker) override {
        Node::restart(treeThinker);
    }
};

class NotFiring : public Node {
    CSGOId sourceId;

public:
    NotFiring(Blackboard & blackboard, CSGOId sourceId) :
            Node(blackboard, "NotFiringNode"), sourceId(sourceId) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & sourceClient = state.getClient(sourceId);

        if (!blackboard.playerToAction[sourceId].getButton(IN_ATTACK)) {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        return playerNodeState[treeThinker.csgoId];
    }

    virtual void restart(const TreeThinker & treeThinker) override {
        Node::restart(treeThinker);
    }
};

class MemoryCheck : public Script {
public:
    MemoryCheck(const ServerState & state) :
            Script("MemoryCheck", {{0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_CT}}, {ObserveType::FirstPerson, 0}) { }

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
                                                                        make_unique<SetPos>(blackboard, Vec3({1440.059204, 1112.913574, -8.766550}), Vec2({90., 0.})),
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
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1031.611084, 962.737915, -0.588848}), Vec2({90., 0.})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0)),
                                                                "MemorySetup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id}, false)
            ), "MemoryDisableDuringSetup");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                //make_unique<AimingAt>(blackboard, neededBots[0].id, neededBots[1].id),
                                                                                                //make_unique<NotFiring>(blackboard, neededBots[0].id),
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[1].id}),
                                                                                                make_unique<movement::WaitNode>(blackboard, 200.5)),
                                                                                        "MemoryCondition")),
                                                 "MemorySequence");
        }
    }
};

#endif //CSKNOW_MEMORY_H
