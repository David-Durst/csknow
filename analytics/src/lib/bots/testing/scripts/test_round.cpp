//
// Created by durst on 4/27/23.
//
#include "bots/testing/scripts/test_round.h"

RoundScript::RoundScript(const csknow::plant_states::PlantStatesResult & plantStatesResult, size_t plantStateIndex) :
    Script("RoundScript", {}, {ObserveType::FirstPerson, 0}) {
    name += std::to_string(plantStateIndex);
    int numCT = 0, numT = 0;
    neededBots.clear();
    for (size_t i = 0; i < csknow::plant_states::max_players_per_team; i++) {
        if (plantStatesResult.ctPlayerStates[i].alive[plantStateIndex] && numCT < maxCT) {
            numCT++;
            neededBots.push_back({0, ENGINE_TEAM_CT});
            playerPos.push_back(plantStatesResult.ctPlayerStates[i].pos[plantStateIndex]);
            playerViewAngle.push_back(plantStatesResult.ctPlayerStates[i].viewAngle[plantStateIndex]);
        }
        if (plantStatesResult.tPlayerStates[i].alive[plantStateIndex] && numT < maxT) {
            numT++;
            neededBots.push_back({0, ENGINE_TEAM_T});
            playerPos.push_back(plantStatesResult.tPlayerStates[i].pos[plantStateIndex]);
            playerViewAngle.push_back(plantStatesResult.tPlayerStates[i].viewAngle[plantStateIndex]);
        }
    }
}

void RoundScript::initialize(Tree &tree, ServerState &state) {
    if (tree.newBlackboard) {
        Blackboard &blackboard = *tree.blackboard;
        Script::initialize(tree, state);
        vector<CSGOId> neededBotIds = getNeededBotIds();
        Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
            make_unique<InitGameRound>(blackboard, name),
            make_unique<movement::WaitNode>(blackboard, 0.3),
            make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
            make_unique<SlayAllBut>(blackboard, neededBotIds, state),
            make_unique<TeleportMultiple>(blackboard, neededBotIds, playerPos, playerViewAngle, state),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
            make_unique<RecomputeOrdersNode>(blackboard)), "RoundSetup");
        Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
            std::move(setupCommands),
            make_unique<DisableActionsNode>(blackboard, "DisableSetup", neededBotIds, false)
        ), "DefuserDisableDuringSetup");

        commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                 std::move(disableAllBothDuringSetup),
                                                 make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                    make_unique<RepeatDecorator>(blackboard, make_unique<RoundStart>(blackboard), true),
                                                                                    make_unique<movement::WaitNode>(blackboard, 60, false)),
                                                                                "RoundCondition")),
                                             "RoundSequence");
    }
}