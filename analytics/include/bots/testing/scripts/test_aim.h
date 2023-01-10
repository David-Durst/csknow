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

    NodeState exec([[maybe_unused]] const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.streamingManager.streamingEngagementAim.printAimTicks = numTicks;
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return NodeState::Success;
    }
};

struct ResetAimController : public Node {
    ResetAimController(Blackboard & blackboard) : Node(blackboard, "ResetAimController") { }

    NodeState exec([[maybe_unused]] const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.streamingManager.forceReset = true;
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
                                                         make_unique<SetPos>(blackboard, Vec3({1440.059204, 1112.913574, 10.766550}), Vec2({-178.035400, -1.805965})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, 10.775691}), Vec2({-89.683349, 0.746031})),
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
                                                                                                make_unique<movement::WaitNode>(blackboard, 4, false)),
                                                                                        "AimAndKillCondition")),
                                                 "AimAndKillSequence");
        }
    }
};

// x range for left to right - 1590 (farthest from b), 1269 (closest to b
// bottom of ramp, middle 1436.137695 2271.818848 -2.052487
// top of ramp, middle 1401.966309 2774.590088 110.369415
namespace variable_aim_test {
    enum class EnemyPos {
        Close,
        BottomRamp,
        TopRamp
    };

    enum class EnemyMovement {
        None,
        Forward,
        Left,
        Right
    };

    enum class AttackerInitialViewAngle {
        HardLeft,
        MidLeft,
        MidRightUp,
        MidRightDown
    };

    class VariableAimAndKillWithinTimeCheck : public Script {
        Vec3 closeRightPos{1593.96875, 1653.840576, 10.489006},
            closeCenterPos{1417.528564, 1653.840576, 10.439991},
            closeLeftPos{1252.031250, 1653.840576, 10.439991},
            midRightPos{1614.279175, 2214.528076, 12.016434},
            midCenterPos{1426.578857, 2214.528076, 10.867785},
            midLeftPos{1249.070068, 2214.528076, 16.434539},
            farRightPos{1604.709351, 2747.997314, 124.082062},
            farCenterPos{1418.727417, 2747.997314, 118.439018},
            farLeftPos{1260.341064, 2747.997314, 139.031250};
        Vec2 hardLeftViewAngle{-179., 0.},
            midLeftViewAngle{110., 0.},
            midRightUpViewAngle{56., -20.},
            midRightDownViewAngle{56., 20.};
        Vec3 pos;
        Vec2 viewAngle;
        int32_t inputBits;
        bool humanAttacker;


    public:
        explicit VariableAimAndKillWithinTimeCheck(EnemyPos enemyPos, EnemyMovement enemyMovement,
                                                   AttackerInitialViewAngle attackerInitialViewAngle, bool humanAttacker) :
                                                   Script("VariableAim", {{0, ENGINE_TEAM_T, AggressiveType::Push, humanAttacker },
                                                                          {0, ENGINE_TEAM_CT}},
                                                          {humanAttacker ? ObserveType::None : ObserveType::FirstPerson, 0}),
                                                   humanAttacker(humanAttacker) {
            switch (enemyPos) {
                case EnemyPos::Close:
                    switch (enemyMovement) {
                        case EnemyMovement::None:
                            name += "CloseNone";
                            pos = closeCenterPos;
                            inputBits = 0;
                            break;
                        case EnemyMovement::Forward:
                            name += "CloseForward";
                            pos = closeCenterPos;
                            inputBits = IN_FORWARD;
                            break;
                        case EnemyMovement::Left:
                            name += "CloseLeft";
                            pos = closeLeftPos;
                            inputBits = IN_MOVELEFT;
                            break;
                        case EnemyMovement::Right:
                            name += "CloseRight";
                            pos = closeRightPos;
                            inputBits = IN_MOVERIGHT;
                            break;
                    }
                    break;
                case EnemyPos::BottomRamp:
                    switch (enemyMovement) {
                        case EnemyMovement::None:
                            name += "BottomRampNone";
                            pos = midCenterPos;
                            inputBits = 0;
                            break;
                        case EnemyMovement::Forward:
                            name += "BottomRampForward";
                            pos = midCenterPos;
                            inputBits = IN_FORWARD;
                            break;
                        case EnemyMovement::Left:
                            name += "BottomRampLeft";
                            pos = midLeftPos;
                            inputBits = IN_MOVELEFT;
                            break;
                        case EnemyMovement::Right:
                            name += "BottomRampRight";
                            pos = midRightPos;
                            inputBits = IN_MOVERIGHT;
                            break;
                    }
                    break;
                case EnemyPos::TopRamp:
                    switch (enemyMovement) {
                        case EnemyMovement::None:
                            name += "TopRampNone";
                            pos = farCenterPos;
                            inputBits = 0;
                            break;
                        case EnemyMovement::Forward:
                            name += "TopRampForward";
                            pos = farCenterPos;
                            inputBits = IN_FORWARD;
                            break;
                        case EnemyMovement::Left:
                            name += "TopRampLeft";
                            pos = farLeftPos;
                            inputBits = IN_MOVELEFT;
                            break;
                        case EnemyMovement::Right:
                            name += "TopRampRight";
                            pos = farRightPos;
                            inputBits = IN_MOVERIGHT;
                            break;
                    }
                    break;
            }

            switch (attackerInitialViewAngle) {
                case AttackerInitialViewAngle::HardLeft:
                    name += "HardLeft";
                    viewAngle = hardLeftViewAngle;
                    break;
                case AttackerInitialViewAngle::MidLeft:
                    name += "MidLeft";
                    viewAngle = midLeftViewAngle;
                    break;
                case AttackerInitialViewAngle::MidRightUp:
                    name += "MidRightUp";
                    viewAngle = midRightUpViewAngle;
                    break;
                case AttackerInitialViewAngle::MidRightDown:
                    name += "MidRightDown";
                    viewAngle = midRightDownViewAngle;
                    break;
            }
        }

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
                                                                        make_unique<SetPos>(blackboard, Vec3({1440.059204, 1112.913574, 10.766550}), viewAngle),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, pos, Vec2({-90., 0.})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<RemoveGuns>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<GiveItem>(blackboard, neededBots[0].id, state, "weapon_ak47"),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetCurrentItem>(blackboard, neededBots[0].id, state, "weapon_ak47"),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1)),
                                                                    "VariableAimAndKillWithinTimeCheckSetup");
                Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id})
                ), "VariableAimAndKillDisableDuringSetup");
                Node::Ptr movingPreAct = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id}),
                    make_unique<ForceActionsNode>(blackboard, vector{neededBots[1].id}, inputBits),
                    make_unique<movement::WaitNode>(blackboard, 1.0)
                ), "VariableMovingPreAct");
                Node::Ptr movingAimBufferFill = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id}),
                    make_unique<ForceActionsNode>(blackboard, vector{neededBots[1].id}, inputBits),
                    make_unique<movement::WaitNode>(blackboard, 0.1)
                ), "VariableMovingPreAct");
                commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         //std::move(movingPreAct),
                                                         //make_unique<ResetAimController>(blackboard),
                                                         //std::move(movingAimBufferFill),
                                                         make_unique<SayIf>(blackboard, humanAttacker, "move mouse"),
                                                         make_unique<PrintAim>(blackboard, 128),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                            make_unique<KilledAfterTime>(blackboard, neededBots[0].id, neededBots[1].id, 0.1),
                                                                                            make_unique<ForceActionsNode>(blackboard, vector{neededBots[1].id}, inputBits),
                                                                                            make_unique<movement::WaitNode>(blackboard, 4, false)),
                                                                                        "VariableAimAndKillCondition")),
                                                     "VariableAimAndKillSequence");
            }
        }
    };

}


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
