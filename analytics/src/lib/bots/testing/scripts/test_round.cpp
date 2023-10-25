//
// Created by durst on 4/27/23.
//
#include "bots/testing/scripts/test_round.h"
#include "bots/testing/scripts/learned/log_nodes.h"

RoundScript::RoundScript(const csknow::plant_states::PlantStatesResult & plantStatesResult, size_t plantStateIndex,
                         size_t numRounds, std::mt19937 gen, std::uniform_real_distribution<> dis,
                         std::optional<vector<bool>> playerFreeze) :
    Script("RoundScript", {}, {ObserveType::FirstPerson, 0}),
    plantStateIndex(plantStateIndex), numRounds(numRounds), playerFreeze(playerFreeze) {
    name += std::to_string(plantStateIndex);
    int numCT = 0, numT = 0;
    neededBots.clear();
    c4Pos = plantStatesResult.c4Pos[plantStateIndex];
    for (size_t i = 0; i < csknow::plant_states::max_players_per_team; i++) {
        if (plantStatesResult.ctPlayerStates[i].alive[plantStateIndex] && numCT < maxCT) {
            numCT++;
            neededBots.push_back({0, ENGINE_TEAM_CT, dis(gen) < 0.5 ? AggressiveType::Push : AggressiveType::Bait});
            playerPos.push_back(plantStatesResult.ctPlayerStates[i].pos[plantStateIndex]);
            playerViewAngle.push_back(plantStatesResult.ctPlayerStates[i].viewAngle[plantStateIndex]);
            // need to flip view angle between recording and game
            double tmpX = playerViewAngle.back().x;
            playerViewAngle.back().x = playerViewAngle.back().y;
            playerViewAngle.back().y = tmpX;
        }
        if (plantStatesResult.tPlayerStates[i].alive[plantStateIndex] && numT < maxT) {
            numT++;
            neededBots.push_back({0, ENGINE_TEAM_T, dis(gen) < 0.5 ? AggressiveType::Push : AggressiveType::Bait});
            playerPos.push_back(plantStatesResult.tPlayerStates[i].pos[plantStateIndex]);
            playerViewAngle.push_back(plantStatesResult.tPlayerStates[i].viewAngle[plantStateIndex]);
            // need to flip view angle between recording and game
            double tmpX = playerViewAngle.back().x;
            playerViewAngle.back().x = playerViewAngle.back().y;
            playerViewAngle.back().y = tmpX;
        }
    }
}

void RoundScript::initialize(Tree &tree, ServerState &state) {
    if (tree.newBlackboard) {
        Blackboard &blackboard = *tree.blackboard;
        Script::initialize(tree, state);
        vector<CSGOId> neededBotIds = getNeededBotIds();
        bool lastRound = numRounds == plantStateIndex + 1;
        Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
            make_unique<InitGameRound>(blackboard, name),
            make_unique<SetMaxRounds>(blackboard, lastRound ? 2 : 20, true),
            make_unique<movement::WaitNode>(blackboard, 0.3),
            make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
            make_unique<SlayAllBut>(blackboard, neededBotIds, state),
            make_unique<TeleportMultiple>(blackboard, neededBotIds, playerPos, playerViewAngle, state),
            make_unique<SetPos>(blackboard, c4Pos, Vec2({0., 0.})),
            make_unique<TeleportPlantedC4>(blackboard),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<DamageActive>(blackboard, neededBotIds[0], neededBotIds[0], state),
            make_unique<SetHealth>(blackboard, neededBotIds[0], state, 100),
            make_unique<movement::WaitNode>(blackboard, 0.1),
            make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
            make_unique<RecomputeOrdersNode>(blackboard)), "RoundSetup");
        Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
            std::move(setupCommands),
            make_unique<DisableActionsNode>(blackboard, "DisableSetup", neededBotIds, false)
        ), "DefuserDisableDuringSetup");

        Node::Ptr finishCondition = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                make_unique<RepeatDecorator>(blackboard, make_unique<RoundStart>(blackboard), true),
                make_unique<csknow::tests::learned::FailIfTimeoutEndNode>(blackboard, name, plantStateIndex, numRounds, 60)));

        Node::Ptr roundLogic = make_unique<SequenceNode>(blackboard, Node::makeList(
                make_unique<csknow::tests::learned::StartNode>(blackboard, name, plantStateIndex, numRounds),
                std::move(finishCondition),
                make_unique<csknow::tests::learned::SuccessEndNode>(blackboard, name, plantStateIndex, numRounds))
        );

        Node::Ptr roundLogicWithFreeze;
        if (playerFreeze) {
            vector<CSGOId> freezeIds;
            for (size_t i = 0; i < neededBotIds.size(); i++) {
                if (playerFreeze.value()[i]) {
                    freezeIds.push_back(neededBotIds[i]);
                }
            }

            roundLogicWithFreeze = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                std::move(roundLogic),
                make_unique<DisableActionsNode>(blackboard, "DisableSetup", freezeIds, false)
            ));
        }
        else {
            roundLogicWithFreeze = std::move(roundLogic);
        }

        commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                std::move(disableAllBothDuringSetup),
                std::move(roundLogicWithFreeze)),
        "RoundSequence");
    }
}

WaitUntilScoreScript::WaitUntilScoreScript() : Script("WaitUntilScoreScript", {{0, ENGINE_TEAM_CT}},
                                                      {ObserveType::FirstPerson, 0}) { }

void WaitUntilScoreScript::initialize(Tree & tree, ServerState & state) {
    if (tree.newBlackboard) {
        Blackboard &blackboard = *tree.blackboard;
        Script::initialize(tree, state);
        commands = make_unique<SequenceNode>(blackboard, Node::makeList(
            make_unique<SlayAllBut>(blackboard, vector<CSGOId>{}, state),
            make_unique<RepeatDecorator>(blackboard, make_unique<WaitUntilScoreLessThan>(blackboard, 3), true),
            make_unique<RepeatDecorator>(blackboard, make_unique<RoundStart>(blackboard), true)
        ), "WaitUntilSequence");
    }
}

vector<Script::Ptr> createRoundScripts(const csknow::plant_states::PlantStatesResult & plantStatesResult,
                                       bool quitAtEnd) {
    vector<Script::Ptr> result;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    size_t numRounds = 300;//static_cast<size_t>(plantStatesResult.size);
    for (size_t i = 0; i < numRounds; i++) {
        result.push_back(make_unique<RoundScript>(plantStatesResult, i/*1*//*8*//*12*//*205*/, numRounds, gen, dis,
                                                  std::nullopt));
    }
    if (quitAtEnd) {
        result.push_back(make_unique<QuitScript>());
    }
    else {
        result.push_back(make_unique<WaitUntilScoreScript>());
    }

    return result;
}

void addRow(csknow::plant_states::PlantStatesResult & plantStatesResult, Vec3 c4Pos) {
    plantStatesResult.c4Pos.push_back(c4Pos);
    for (size_t i = 0; i < csknow::plant_states::max_players_per_team; i++) {
        plantStatesResult.ctPlayerStates[i].alive.push_back(false);
        plantStatesResult.ctPlayerStates[i].pos.push_back({});
        plantStatesResult.ctPlayerStates[i].viewAngle.push_back({});
        plantStatesResult.tPlayerStates[i].alive.push_back(false);
        plantStatesResult.tPlayerStates[i].pos.push_back({});
        plantStatesResult.tPlayerStates[i].viewAngle.push_back({});
    }
}

void repeatRow(csknow::plant_states::PlantStatesResult & plantStatesResult, vector<vector<bool>> & playerFreeze,
               int numTimes) {
    for (int i = 0; i < numTimes; i++) {
        addRow(plantStatesResult, plantStatesResult.c4Pos.back());
        size_t newIndex = plantStatesResult.ctPlayerStates[i].alive.size() - 1;
        for (size_t i = 0; i < csknow::plant_states::max_players_per_team; i++) {
            plantStatesResult.ctPlayerStates[i].alive[newIndex] = plantStatesResult.ctPlayerStates[i].alive[newIndex-1];
            plantStatesResult.ctPlayerStates[i].pos[newIndex] = plantStatesResult.ctPlayerStates[i].pos[newIndex-1];
            plantStatesResult.ctPlayerStates[i].viewAngle[newIndex] = plantStatesResult.ctPlayerStates[i].viewAngle[newIndex-1];
            plantStatesResult.tPlayerStates[i].alive[newIndex] = plantStatesResult.tPlayerStates[i].alive[newIndex-1];
            plantStatesResult.tPlayerStates[i].pos[newIndex] = plantStatesResult.tPlayerStates[i].pos[newIndex-1];
            plantStatesResult.tPlayerStates[i].viewAngle[newIndex] = plantStatesResult.tPlayerStates[i].viewAngle[newIndex-1];
        }
        playerFreeze.push_back(playerFreeze.back());
    }

}

vector<Script::Ptr> createPrebakedRoundScripts(bool quitAtEnd) {
    vector<Script::Ptr> result;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    int numRepeats = 9;

    vector<vector<bool>> playerFreeze;
    csknow::plant_states::PlantStatesResult plantStatesResult;
    // attack a from spawn, t on site and in mid doors
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {-1006.737915, 2148.301025, -31.970458};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {-3.328921, 0.309759};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {1160.000976, 2573.304931, 96.338958};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-144., 1.084169};
    plantStatesResult.tPlayerStates[1].alive.back() = true;
    plantStatesResult.tPlayerStates[1].pos.back() = {-362.730529, 1595.130981, -126.807861};
    plantStatesResult.tPlayerStates[1].viewAngle.back() = {94.862510, -2.129631};
    playerFreeze.push_back({false, true, true, false, false,
                            false, false, false, false, false});
    repeatRow(plantStatesResult, playerFreeze, numRepeats);
    // defend a against cat
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {402.177368, 1875.845092, 95.393173};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {43.959850, 3.755849};
    plantStatesResult.ctPlayerStates[1].alive.back() = true;
    plantStatesResult.ctPlayerStates[1].pos.back() = {357.684234, 1650.239990, 27.671302};
    plantStatesResult.ctPlayerStates[1].viewAngle.back() = {71.024917, -9.370210};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {1160.000976, 2573.304931, 96.338958};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-144., 1.084169};
    playerFreeze.push_back({true, false, true, false, false,
                            false, false, false, false, false});
    repeatRow(plantStatesResult, playerFreeze, numRepeats);
    // defend a against ramp
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {1393.406738, 521.030822, -94.765136};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {91.973045, -5.304626};
    plantStatesResult.ctPlayerStates[1].alive.back() = true;
    plantStatesResult.ctPlayerStates[1].pos.back() = {1266.489990, 1308.994018, 0.008083};
    plantStatesResult.ctPlayerStates[1].viewAngle.back() = {89.746215, -3.446030};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {1160.000976, 2573.304931, 96.338958};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-144., 1.084169};
    playerFreeze.push_back({true, false, true, false, false,
                            false, false, false, false, false});
    repeatRow(plantStatesResult, playerFreeze, numRepeats);
    // defend b against site
    addRow(plantStatesResult, {-1427.551391, 2500.479492, 2.367282});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {-1445.885375, 2497.657958, 1.294036};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {4.949440, -126.222084};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {-1977.860229, 1665.813110, 31.853256};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-19.819931, 3.903996};
    playerFreeze.push_back({true, false, false, false, false,
                            false, false, false, false, false});
    repeatRow(plantStatesResult, playerFreeze, numRepeats);
    // defend b against tuns
    addRow(plantStatesResult, {-1427.551391, 2500.479492, 2.367282});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {-1078.543823, 1232.906372, -87.452003};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {4.949440, -126.222084};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {-1977.860229, 1665.813110, 31.853256};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-19.819931, 3.903996};
    playerFreeze.push_back({true, false, false, false, false,
                            false, false, false, false, false});
    repeatRow(plantStatesResult, playerFreeze, numRepeats);
    plantStatesResult.size = plantStatesResult.ctPlayerStates[0].alive.size();

    size_t numRounds = static_cast<size_t>(plantStatesResult.size);
    for (size_t i = 0; i < numRounds; i++) {
        result.push_back(make_unique<RoundScript>(plantStatesResult, i/*1*//*8*//*12*//*205*/, numRounds, gen, dis,
                                                  playerFreeze[i]));
    }
    if (quitAtEnd) {
        result.push_back(make_unique<QuitScript>());
    }
    else {
        result.push_back(make_unique<WaitUntilScoreScript>());
    }

    return result;
}
