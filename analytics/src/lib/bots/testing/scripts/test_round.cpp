//
// Created by durst on 4/27/23.
//
#include "bots/testing/scripts/test_round.h"
#include "bots/testing/scripts/learned/log_nodes.h"

RoundScript::RoundScript(const csknow::plant_states::PlantStatesResult & plantStatesResult, size_t plantStateIndex,
                         size_t numRounds, std::mt19937 gen, std::uniform_real_distribution<> dis,
                         std::optional<vector<bool>> playerFreeze, string baseName) :
    Script(baseName, {}, {ObserveType::FirstPerson, 0}),
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
                                                  std::nullopt, "RoundScript"));
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
               vector<string> & names, int numRepeats) {
    for (int r = 0; r < numRepeats; r++) {
        addRow(plantStatesResult, plantStatesResult.c4Pos.back());
        size_t newIndex = plantStatesResult.ctPlayerStates[0].alive.size() - 1;
        for (size_t i = 0; i < csknow::plant_states::max_players_per_team; i++) {
            plantStatesResult.ctPlayerStates[i].alive[newIndex] = plantStatesResult.ctPlayerStates[i].alive[newIndex-1];
            plantStatesResult.ctPlayerStates[i].pos[newIndex] = plantStatesResult.ctPlayerStates[i].pos[newIndex-1];
            plantStatesResult.ctPlayerStates[i].viewAngle[newIndex] = plantStatesResult.ctPlayerStates[i].viewAngle[newIndex-1];
            plantStatesResult.tPlayerStates[i].alive[newIndex] = plantStatesResult.tPlayerStates[i].alive[newIndex-1];
            plantStatesResult.tPlayerStates[i].pos[newIndex] = plantStatesResult.tPlayerStates[i].pos[newIndex-1];
            plantStatesResult.tPlayerStates[i].viewAngle[newIndex] = plantStatesResult.tPlayerStates[i].viewAngle[newIndex-1];
        }
        playerFreeze.push_back(playerFreeze.back());
        names.push_back(names.back());
    }

}

vector<Script::Ptr> createPrebakedRoundScripts(bool quitAtEnd) {
    vector<Script::Ptr> result;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    int numRepeats = 29;

    vector<vector<bool>> playerFreeze;
    vector<string> names;
    csknow::plant_states::PlantStatesResult plantStatesResult;
    // attack a from spawn, need to eliminate t hiding long
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {1430.616699, 1916.052490, -10.300033};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {112.955604, -4.299486};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {1704.018188, 1011.443786, 2.233371};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-1.860130, -178.045181};
    playerFreeze.push_back({false, true, false, false, false,
                            false, false, false, false, false});
    names.emplace_back("AttackASpawnTLong");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    // attack a from spawn, need to eliminate t hiding extendedA
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {1430.616699, 1916.052490, -10.300033};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {112.955604, -4.299486};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {563.968750, 2759.416259, 97.259826};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-45.278255, 1.510083};
    playerFreeze.push_back({false, true, false, false, false,
                            false, false, false, false, false});
    names.emplace_back("AttackASpawnTExtendedA");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    // attack a from cat, no teammate
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {308.136962, 1628.022460, 12.358312};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {66.387672, -17.423997};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {1051.031250, 2939.113281, 128.593978};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {53.029331, -3.833290};
    playerFreeze.push_back({false, true, false, false, false,
                            false, false, false, false, false});
    names.emplace_back("AttackACatNoTeammate");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    // attack a from cat, teammate cat covering
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {308.136962, 1628.022460, 12.358312};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {66.387672, -17.423997};
    plantStatesResult.ctPlayerStates[1].alive.back() = true;
    plantStatesResult.ctPlayerStates[1].pos.back() = {462.430969, 2006.059082, 133.031250};
    plantStatesResult.ctPlayerStates[1].viewAngle.back() = {42.536399, 2.168326};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {1051.031250, 2939.113281, 128.593978};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {53.029331, -3.833290};
    playerFreeze.push_back({false, true, true, false, false,
                            false, false, false, false, false});
    names.emplace_back("AttackACatOneTeammate");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    // attack a from, two teammates cat covering
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {308.136962, 1628.022460, 12.358312};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {66.387672, -17.423997};
    plantStatesResult.ctPlayerStates[1].alive.back() = true;
    plantStatesResult.ctPlayerStates[1].pos.back() = {462.430969, 2006.059082, 133.031250};
    plantStatesResult.ctPlayerStates[1].viewAngle.back() = {42.536399, 2.168326};
    plantStatesResult.ctPlayerStates[2].alive.back() = true;
    plantStatesResult.ctPlayerStates[2].pos.back() = {462.430969, 2056.059082, 133.031250};
    plantStatesResult.ctPlayerStates[2].viewAngle.back() = {42.536399, 2.168326};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {1051.031250, 2939.113281, 128.593978};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {53.029331, -3.833290};
    playerFreeze.push_back({false, true, true, true, false,
                            false, false, false, false, false});
    names.emplace_back("AttackACatTwoTeammates");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    // attack b hole, teammate b doors
    addRow(plantStatesResult, {-1427.551391, 2500.479492, 2.367282});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {-583.364318, 2389.586425, -100.925903};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {178.822967, -11.732166};
    plantStatesResult.ctPlayerStates[1].alive.back() = true;
    plantStatesResult.ctPlayerStates[1].pos.back() = {-1396.848022, 2144.354980, 1.107921};
    plantStatesResult.ctPlayerStates[1].viewAngle.back() = {-165.303222, -0.464639};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    //plantStatesResult.tPlayerStates[0].pos.back() = {-1959.919799, 1532.453491, 33.999443};
    plantStatesResult.tPlayerStates[0].pos.back() = {-1879.674072, 2378.484130, 8.714675};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {89.175971, 0.380478};
    playerFreeze.push_back({false, true, true, false, false,
                            false, false, false, false, false});
    names.emplace_back("AttackBHoleTeammateBDoors");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    // attack b doors, teammate hole
    addRow(plantStatesResult, {-1427.551391, 2500.479492, 2.367282});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {-550.731201, 2076.939208, -118.991142};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {178.822967, -11.732166};
    plantStatesResult.ctPlayerStates[1].alive.back() = true;
    plantStatesResult.ctPlayerStates[1].pos.back() = {-1395.869873, 2652.096679, 125.027893};
    plantStatesResult.ctPlayerStates[1].viewAngle.back() = {-157.395126, 16.920925};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {-1879.674072, 2378.484130, 8.714675};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {89.175971, 0.380478};
    playerFreeze.push_back({false, true, true, false, false,
                            false, false, false, false, false});
    names.emplace_back("AttackBDoorsTeammateHole");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
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
    names.emplace_back("DefendACTCat");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    // defend a against long
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
    names.emplace_back("DefendACTLong");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
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
    names.emplace_back("DefendBCTSite");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
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
    names.emplace_back("DefendBCTTuns");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    plantStatesResult.size = plantStatesResult.ctPlayerStates[0].alive.size();

    size_t numRounds = static_cast<size_t>(plantStatesResult.size);
    for (size_t i = 0; i < numRounds; i++) {
        result.push_back(make_unique<RoundScript>(plantStatesResult, i/*1*//*8*//*12*//*205*/, numRounds, gen, dis,
                                                  playerFreeze[i], names[i]));
    }
    if (quitAtEnd) {
        result.push_back(make_unique<QuitScript>());
    }
    else {
        result.push_back(make_unique<WaitUntilScoreScript>());
    }

    return result;
}
