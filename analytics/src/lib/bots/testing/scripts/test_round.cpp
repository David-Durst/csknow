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

void randomizePositions(csknow::plant_states::PlantStatesResult & plantStatesResult, const nav_mesh::nav_file & navFile,
                        std::mt19937 gen, std::uniform_real_distribution<> dis) {
    for (size_t row = 0; row < plantStatesResult.ctPlayerStates[0].alive.size(); row++) {
        for (size_t i = 0; i < csknow::plant_states::max_players_per_team; i++) {
            if (plantStatesResult.ctPlayerStates[i].alive[row]) {
                plantStatesResult.ctPlayerStates[i].pos[row].x += (dis(gen) - 0.5) * WIDTH;
                plantStatesResult.ctPlayerStates[i].pos[row].y += (dis(gen) - 0.5) * WIDTH;
                Vec3 newPosPreFit = plantStatesResult.ctPlayerStates[i].pos[row];
                AreaId newAreaId = navFile.get_nearest_area_by_position(vec3Conv(newPosPreFit)).get_id();
                plantStatesResult.ctPlayerStates[i].pos[row] = vec3tConv(navFile.get_nearest_point_in_area(
                        vec3Conv(newPosPreFit),
                        navFile.get_area_by_id_fast(newAreaId))
                );
            }
            if (plantStatesResult.tPlayerStates[i].alive[row]) {
                plantStatesResult.tPlayerStates[i].pos[row].x += (dis(gen) - 0.5) * WIDTH;
                plantStatesResult.tPlayerStates[i].pos[row].y += (dis(gen) - 0.5) * WIDTH;
                Vec3 newPosPreFit = plantStatesResult.tPlayerStates[i].pos[row];
                AreaId newAreaId = navFile.get_nearest_area_by_position(vec3Conv(newPosPreFit)).get_id();
                plantStatesResult.tPlayerStates[i].pos[row] = vec3tConv(navFile.get_nearest_point_in_area(
                        vec3Conv(newPosPreFit),
                        navFile.get_area_by_id_fast(newAreaId))
                );
            }
        }
    }
}

vector<Script::Ptr> createPrebakedRoundScripts(const nav_mesh::nav_file & navFile, bool randomizePositions,
                                               bool quitAtEnd) {
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
    plantStatesResult.ctPlayerStates[0].pos.back() = {1430.616699, 1816.052490, -10.300033};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {112.955604, -4.299486};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {1704.018188, 1011.443786, 2.233371};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-1.860130, -178.045181};
    playerFreeze.push_back({false, true, false, false, false,
                            false, false, false, false, false});
    names.emplace_back("AttackASpawnTLong");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    // attack a from spawn, need to eliminate t hiding long, teammates covering that enemy
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {1430.616699, 1816.052490, -10.300033};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {112.955604, -4.299486};
    plantStatesResult.ctPlayerStates[1].alive.back() = true;
    plantStatesResult.ctPlayerStates[1].pos.back() = {1430.616699, 1516.052490, -10.300033};
    plantStatesResult.ctPlayerStates[1].viewAngle.back() = {112.955604, -4.299486};
    plantStatesResult.ctPlayerStates[2].alive.back() = true;
    plantStatesResult.ctPlayerStates[2].pos.back() = {1430.616699, 1316.052490, -10.300033};
    plantStatesResult.ctPlayerStates[2].viewAngle.back() = {112.955604, -4.299486};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {1704.018188, 1011.443786, 2.233371};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-1.860130, -178.045181};
    playerFreeze.push_back({false, true, true, true, false,
                            false, false, false, false, false});
    names.emplace_back("AttackASpawnTLongTwoTeammates");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    // attack a from spawn, need to eliminate t hiding extendedA
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {1430.616699, 1816.052490, -10.300033};
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
    // attack a from cat, two teammates cat covering
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {308.136962, 1628.022460, 12.358312};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {66.387672, -17.423997};
    plantStatesResult.ctPlayerStates[1].alive.back() = true;
    plantStatesResult.ctPlayerStates[1].pos.back() = {462.430969, 2006.059082, 133.031250};
    plantStatesResult.ctPlayerStates[1].viewAngle.back() = {42.536399, 2.168326};
    plantStatesResult.ctPlayerStates[2].alive.back() = true;
    plantStatesResult.ctPlayerStates[2].pos.back() = {563.968750, 2763.999511, 97.379516};
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
    plantStatesResult.ctPlayerStates[0].pos.back() = {-550.731201, 2076.939208, -118.991142};
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
    plantStatesResult.ctPlayerStates[0].pos.back() = {563.968750, 2763.999511, 97.379516};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {-89.047363, 1.806404};
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
    // defend a against cat teammates covering behind
    addRow(plantStatesResult, {1241., 2586., 127.});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {563.968750, 2763.999511, 97.379516};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {-89.047363, 1.806404};
    plantStatesResult.ctPlayerStates[1].alive.back() = true;
    plantStatesResult.ctPlayerStates[1].pos.back() = {357.684234, 1650.239990, 27.671302};
    plantStatesResult.ctPlayerStates[1].viewAngle.back() = {71.024917, -9.370210};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {1160.000976, 2573.304931, 96.338958};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-144., 1.084169};
    plantStatesResult.tPlayerStates[1].alive.back() = true;
    plantStatesResult.tPlayerStates[1].pos.back() = {1175.846923, 2944.958984, 128.266784};
    plantStatesResult.tPlayerStates[1].viewAngle.back() = {-127.956420, 1.114561};
    plantStatesResult.tPlayerStates[2].alive.back() = true;
    plantStatesResult.tPlayerStates[2].pos.back() = {1427.594238, 2308.249023, 4.196350};
    plantStatesResult.tPlayerStates[2].viewAngle.back() = {-165.436294, -4.732160};
    playerFreeze.push_back({true, false, true, true, true,
                            false, false, false, false, false});
    names.emplace_back("DefendACTCatTwoTeammates");
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
    // defend b against hole
    addRow(plantStatesResult, {-1427.551391, 2500.479492, 2.367282});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {-1179.737426, 2664.458007, 79.098220};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {176.255645, -1.181761};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {-1430.002441, 2676.153564, 16.374132};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-4.951731, -15.823047};
    playerFreeze.push_back({true, false, false, false, false,
                            false, false, false, false, false});
    names.emplace_back("DefendBCTHole");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    // defend b against hole, two teammates to keep in place
    addRow(plantStatesResult, {-1427.551391, 2500.479492, 2.367282});
    plantStatesResult.ctPlayerStates[0].alive.back() = true;
    plantStatesResult.ctPlayerStates[0].pos.back() = {-1179.737426, 2664.458007, 79.098220};
    plantStatesResult.ctPlayerStates[0].viewAngle.back() = {176.255645, -1.181761};
    plantStatesResult.tPlayerStates[0].alive.back() = true;
    plantStatesResult.tPlayerStates[0].pos.back() = {-1430.002441, 2676.153564, 16.374132};
    plantStatesResult.tPlayerStates[0].viewAngle.back() = {-4.951731, -15.823047};
    plantStatesResult.tPlayerStates[1].alive.back() = true;
    plantStatesResult.tPlayerStates[1].pos.back() = {-1925.693725, 2991.133300, 36.464263};
    plantStatesResult.tPlayerStates[1].viewAngle.back() = {-56.154346, -2.903999};
    plantStatesResult.tPlayerStates[2].alive.back() = true;
    plantStatesResult.tPlayerStates[2].pos.back() = {-1898.840698, 2345.118164, 3.255815};
    plantStatesResult.tPlayerStates[2].viewAngle.back() = {23.841018, -5.536960};
    playerFreeze.push_back({true, false, true, true, false,
                            false, false, false, false, false});
    names.emplace_back("DefendBCTHoleTwoTeammates");
    repeatRow(plantStatesResult, playerFreeze, names, numRepeats);
    plantStatesResult.size = plantStatesResult.ctPlayerStates[0].alive.size();

    randomizePositions(plantStatesResult, navFile, gen, dis);

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
