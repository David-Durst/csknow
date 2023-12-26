//
// Created by durst on 4/27/23.
//
#include "bots/testing/scripts/test_round.h"
#include "bots/testing/scripts/learned/log_nodes.h"

RoundScript::RoundScript(const csknow::plant_states::PlantStatesResult & plantStatesResult, size_t plantStateIndex,
                         size_t numRounds, std::mt19937 gen, std::uniform_real_distribution<> dis,
                         std::optional<vector<bool>> playerFreeze, string baseName, std::optional<Vec3> cameraOrigin,
                         std::optional<Vec2> cameraAngle, int numHumans) :
    Script(baseName, {}, {ObserveType::FirstPerson, 0}),
    plantStateIndex(plantStateIndex), numRounds(numRounds), playerFreeze(playerFreeze) {
    name += std::to_string(plantStateIndex);
    if (cameraOrigin) {
        observeSettings.observeType = ObserveType::Absolute;
        observeSettings.cameraOrigin = cameraOrigin.value();
        // this has to be flipped (done in code below when creating situations) because being passed directly
        observeSettings.cameraAngle = cameraAngle.value();
    }
    else if (!plantStatesResult.cameraPos.empty()) {
        observeSettings.observeType = ObserveType::Absolute;
        observeSettings.cameraOrigin = plantStatesResult.cameraPos[plantStateIndex];
        // this has to be flipped because being passed directly
        observeSettings.cameraAngle = plantStatesResult.cameraViewAngle[plantStateIndex];
        double tmpX = observeSettings.cameraAngle.x;
        observeSettings.cameraAngle.x = observeSettings.cameraAngle.y;
        observeSettings.cameraAngle.y = tmpX;
    }
    int numCT = 0, numT = 0;
    neededBots.clear();
    c4Pos = plantStatesResult.c4Pos[plantStateIndex];
    for (size_t i = 0; i < csknow::plant_states::max_players_per_team; i++) {
        if (plantStatesResult.ctPlayerStates[i].alive[plantStateIndex] && numCT < maxCT) {
            neededBots.push_back({0, ENGINE_TEAM_CT, dis(gen) < 0.5 ? AggressiveType::Push : AggressiveType::Bait});
            playerPos.push_back(plantStatesResult.ctPlayerStates[i].pos[plantStateIndex]);
            playerViewAngle.push_back(plantStatesResult.ctPlayerStates[i].viewAngle[plantStateIndex]);
            // no need to flip view angle between recording and game, my bt test set pos command already does it
            health.push_back(static_cast<int>(100 * plantStatesResult.ctPlayerStates[i].health[plantStateIndex]));
            armor.push_back(static_cast<int>(100 * plantStatesResult.ctPlayerStates[i].armor[plantStateIndex]));
            helmet.push_back(plantStatesResult.ctPlayerStates[i].helmet[plantStateIndex]);
            numCT++;
        }
        if (plantStatesResult.tPlayerStates[i].alive[plantStateIndex] && numT < maxT) {
            neededBots.push_back({0, ENGINE_TEAM_T, dis(gen) < 0.5 ? AggressiveType::Push : AggressiveType::Bait});
            if (numT < numHumans) {
                neededBots.back().human = true;
            }
            playerPos.push_back(plantStatesResult.tPlayerStates[i].pos[plantStateIndex]);
            playerViewAngle.push_back(plantStatesResult.tPlayerStates[i].viewAngle[plantStateIndex]);
            // no need to flip view angle between recording and game, my bt test set pos command already does it
            health.push_back(static_cast<int>(100 * plantStatesResult.tPlayerStates[i].health[plantStateIndex]));
            armor.push_back(static_cast<int>(100 * plantStatesResult.tPlayerStates[i].armor[plantStateIndex]));
            helmet.push_back(plantStatesResult.tPlayerStates[i].helmet[plantStateIndex]);
            numT++;
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
            make_unique<SetHealthArmorHelmetMultiple>(blackboard, neededBotIds, health, armor, helmet, state),
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
                                       int startSituationId, bool quitAtEnd, int numHumans) {
    vector<Script::Ptr> result;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    size_t numRounds = static_cast<size_t>(plantStatesResult.size);

    size_t batchSize = 100;
    size_t maxI = std::min(batchSize * (startSituationId + 1), numRounds);
    if (startSituationId == -1) {
        startSituationId = 0;
        maxI = numRounds;
    }
    for (size_t i = batchSize * startSituationId; i < maxI; i++) {
        // 0 - push in to b from mid with enemy in b site
        // 4 - attacking a from cat and spawn
        result.push_back(make_unique<RoundScript>(plantStatesResult, i/*0*//*4*//*8*//*12*//*205*/, maxI, gen, dis,
                                                  std::nullopt, "RoundScript", std::nullopt, std::nullopt,
                                                  numHumans));
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
        plantStatesResult.ctPlayerStates[i].health.push_back(1.);
        plantStatesResult.ctPlayerStates[i].armor.push_back(1.);
        plantStatesResult.ctPlayerStates[i].helmet.push_back(true);
        plantStatesResult.tPlayerStates[i].alive.push_back(false);
        plantStatesResult.tPlayerStates[i].pos.push_back({});
        plantStatesResult.tPlayerStates[i].viewAngle.push_back({});
        plantStatesResult.tPlayerStates[i].health.push_back(1.);
        plantStatesResult.tPlayerStates[i].armor.push_back(1.);
        plantStatesResult.tPlayerStates[i].helmet.push_back(true);
    }
}

void repeatRow(csknow::plant_states::PlantStatesResult & plantStatesResult, vector<vector<bool>> & playerFreeze,
               vector<string> & names, vector<Vec3> & cameraPoses, vector<Vec2> & cameraAngles, int numRepeats) {
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
        cameraPoses.push_back(cameraPoses.back());
        cameraAngles.push_back(cameraAngles.back());
    }

}

Vec3 getPosInNavMesh(Vec3 origPos, Vec3 targetPos, const nav_mesh::nav_file & navFile) {
    // camera to foot position
    origPos.z -= 64.;
    const nav_mesh::nav_area & origArea = navFile.get_nearest_area_by_position(vec3Conv(origPos));
    AABB origAreaAABB{vec3tConv(origArea.get_min_corner()), vec3tConv(origArea.get_max_corner())};
    Vec3 endPos = vec3tConv(navFile.get_nearest_point_in_area(
            vec3Conv(targetPos),
            origArea)
    );
    if (origAreaAABB.max.x - origAreaAABB.min.x < WIDTH) {
        endPos.x = getCenter(origAreaAABB).x;
    }
    else {
        endPos.x = std::min(origAreaAABB.max.x - WIDTH * 0.5, endPos.x);
        endPos.x = std::max(origAreaAABB.min.x + WIDTH * 0.5, endPos.x);
    }
    if (origAreaAABB.max.y - origAreaAABB.min.y < WIDTH) {
        endPos.y = getCenter(origAreaAABB).y;
    }
    else {
        endPos.y = std::min(origAreaAABB.max.y - WIDTH * 0.5, endPos.y);
        endPos.y = std::max(origAreaAABB.min.y + WIDTH * 0.5, endPos.y);
    }
    endPos.z += 10.;
    return endPos;
}

void randomizePositions(csknow::plant_states::PlantStatesResult & plantStatesResult, const nav_mesh::nav_file & navFile,
                        std::mt19937 gen, std::uniform_real_distribution<> dis) {
    for (size_t row = 0; row < plantStatesResult.ctPlayerStates[0].alive.size(); row++) {
        for (size_t i = 0; i < csknow::plant_states::max_players_per_team; i++) {
            if (plantStatesResult.ctPlayerStates[i].alive[row]) {
                Vec3 origPos = plantStatesResult.ctPlayerStates[i].pos[row];
                plantStatesResult.ctPlayerStates[i].pos[row].x += (dis(gen) - 0.5) * WIDTH;
                plantStatesResult.ctPlayerStates[i].pos[row].y += (dis(gen) - 0.5) * WIDTH;
                plantStatesResult.ctPlayerStates[i].pos[row] =
                        getPosInNavMesh(origPos, plantStatesResult.ctPlayerStates[i].pos[row], navFile);
            }
            if (plantStatesResult.tPlayerStates[i].alive[row]) {
                Vec3 origPos = plantStatesResult.tPlayerStates[i].pos[row];
                plantStatesResult.tPlayerStates[i].pos[row].x += (dis(gen) - 0.5) * WIDTH;
                plantStatesResult.tPlayerStates[i].pos[row].y += (dis(gen) - 0.5) * WIDTH;
                plantStatesResult.tPlayerStates[i].pos[row] =
                        getPosInNavMesh(origPos, plantStatesResult.tPlayerStates[i].pos[row], navFile);
            }
        }
    }
}

vector<Script::Ptr> createPrebakedRoundScripts(const nav_mesh::nav_file & navFile, bool shouldRandomizePositions,
                                               int situationId, bool quitAtEnd) {
    vector<Script::Ptr> result;

    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

    int numRepeats = 99;

    vector<vector<bool>> playerFreeze;
    vector<string> names;
    vector<Vec3> cameraPoses;
    vector<Vec2> cameraAngles;
    csknow::plant_states::PlantStatesResult plantStatesResult;

    Vec3 aSiteToLongCatCameraPos = {1471.098389, 3066.215576, 498.097443};
    Vec2 aSiteToLongCatCameraAngle = {37.676918, -118.432442};
    Vec3 bSiteToSpawnCameraPos = {-2038.604980, 2418.324707, 985.734863};
    Vec2 bSiteToSpawnCameraAngle = {61.994862, -5.878078};
    Vec3 bUpperTunsToSiteLowerTunsCameraPos = {-2050.380371, 1108.248901, 136.914352};
    Vec2 bUpperTunsToSiteLowerTunsCameraAngle = {10.611423, 43.495888};

    if (situationId == 0 || situationId == -1) {
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
        cameraPoses.push_back(aSiteToLongCatCameraPos);
        cameraAngles.push_back(aSiteToLongCatCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 1 || situationId == -1) {
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
        cameraPoses.push_back(aSiteToLongCatCameraPos);
        cameraAngles.push_back(aSiteToLongCatCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 2 || situationId == -1) {
        // attack a from spawn, need to eliminate t hiding extendedA
        addRow(plantStatesResult, {1241., 2586., 127.});
        plantStatesResult.ctPlayerStates[0].alive.back() = true;
        // enable to get a simpler path straight through long rather than from spawn
        if (false) {
            plantStatesResult.ctPlayerStates[0].pos.back() = {1409.643066, 951.790649, 55.921726};
            plantStatesResult.ctPlayerStates[0].viewAngle.back() = {89.289635, 1.749142};
        }
        else {
            plantStatesResult.ctPlayerStates[0].pos.back() = {1430.616699, 1816.052490, -10.300033};
            plantStatesResult.ctPlayerStates[0].viewAngle.back() = {112.955604, -4.299486};
        }
        plantStatesResult.tPlayerStates[0].alive.back() = true;
        plantStatesResult.tPlayerStates[0].pos.back() = {563.968750, 2759.416259, 97.259826};
        plantStatesResult.tPlayerStates[0].viewAngle.back() = {-45.278255, 1.510083};
        playerFreeze.push_back({false, true, false, false, false,
                                false, false, false, false, false});
        names.emplace_back("AttackASpawnTExtendedA");
        cameraPoses.push_back(aSiteToLongCatCameraPos);
        cameraAngles.push_back(aSiteToLongCatCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 3 || situationId == -1) {
        // attack b hole, teammate b doors
        addRow(plantStatesResult, {-1427.551391, 2500.479492, 2.367282});
        plantStatesResult.ctPlayerStates[0].alive.back() = true;
        plantStatesResult.ctPlayerStates[0].pos.back() = {-550.731201, 2076.939208, -118.991142};
        plantStatesResult.ctPlayerStates[0].viewAngle.back() = {178.822967, -11.732166};
        plantStatesResult.ctPlayerStates[1].alive.back() = true;
        plantStatesResult.ctPlayerStates[1].pos.back() = {-1396.848022, 2144.354980, 1.107921};
        plantStatesResult.ctPlayerStates[1].viewAngle.back() = {-165.303222, -0.464639};
        plantStatesResult.tPlayerStates[0].alive.back() = true;
        plantStatesResult.tPlayerStates[0].pos.back() = {-1879.674072, 2378.484130, 8.714675};
        plantStatesResult.tPlayerStates[0].viewAngle.back() = {89.175971, 0.380478};
        playerFreeze.push_back({false, true, true, false, false,
                                false, false, false, false, false});
        names.emplace_back("AttackBHoleTeammateBDoors");
        cameraPoses.push_back(bSiteToSpawnCameraPos);
        cameraAngles.push_back(bSiteToSpawnCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 4 || situationId == -1) {
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
        cameraPoses.push_back(bSiteToSpawnCameraPos);
        cameraAngles.push_back(bSiteToSpawnCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 5 || situationId == -1) {
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
        plantStatesResult.tPlayerStates[0].viewAngle.back() = {-0.735680, -142.272674};
        playerFreeze.push_back({true, false, true, false, false,
                                false, false, false, false, false});
        names.emplace_back("DefendACTCat");
        cameraPoses.push_back(aSiteToLongCatCameraPos);
        cameraAngles.push_back(aSiteToLongCatCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 6 || situationId == -1) {
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
        cameraPoses.push_back(aSiteToLongCatCameraPos);
        cameraAngles.push_back(aSiteToLongCatCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 7 || situationId == -1) {
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
        cameraPoses.push_back(aSiteToLongCatCameraPos);
        cameraAngles.push_back(aSiteToLongCatCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 8 || situationId == -1) {
        // defend a against long with teammate support
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
        plantStatesResult.tPlayerStates[1].alive.back() = true;
        plantStatesResult.tPlayerStates[1].pos.back() = {1105.557983, 3027.858642, 129.793701};
        plantStatesResult.tPlayerStates[1].viewAngle.back() = {-83.956398, 4.489911};
        playerFreeze.push_back({true, false, true, true, false,
                                false, false, false, false, false});
        names.emplace_back("DefendACTLongWithTeammate");
        cameraPoses.push_back(aSiteToLongCatCameraPos);
        cameraAngles.push_back(aSiteToLongCatCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 9 || situationId == -1) {
        // defend a against long with two teammate support
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
        plantStatesResult.tPlayerStates[1].alive.back() = true;
        plantStatesResult.tPlayerStates[1].pos.back() = {1105.557983, 3027.858642, 129.793701};
        plantStatesResult.tPlayerStates[1].viewAngle.back() = {-83.956398, 4.489911};
        plantStatesResult.tPlayerStates[2].alive.back() = true;
        plantStatesResult.tPlayerStates[2].pos.back() = {1319.525756, 2899.028564, 128.393173};
        plantStatesResult.tPlayerStates[2].viewAngle.back() = {-87.314796, 4.414069};
        playerFreeze.push_back({true, false, true, true, true,
                                false, false, false, false, false});
        names.emplace_back("DefendACTLongWithTwoTeammates");
        cameraPoses.push_back(aSiteToLongCatCameraPos);
        cameraAngles.push_back(aSiteToLongCatCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 10 || situationId == -1) {
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
        cameraPoses.push_back(bUpperTunsToSiteLowerTunsCameraPos);
        cameraAngles.push_back(bUpperTunsToSiteLowerTunsCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 11 || situationId == -1) {
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
        cameraPoses.push_back(bUpperTunsToSiteLowerTunsCameraPos);
        cameraAngles.push_back(bUpperTunsToSiteLowerTunsCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 12 || situationId == -1) {
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
        cameraPoses.push_back(bSiteToSpawnCameraPos);
        cameraAngles.push_back(bSiteToSpawnCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    if (situationId == 13 || situationId == -1) {
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
        cameraPoses.push_back(bSiteToSpawnCameraPos);
        cameraAngles.push_back(bSiteToSpawnCameraAngle);
        repeatRow(plantStatesResult, playerFreeze, names, cameraPoses, cameraAngles, numRepeats);
    }
    plantStatesResult.size = plantStatesResult.ctPlayerStates[0].alive.size();

    if (shouldRandomizePositions) {
        randomizePositions(plantStatesResult, navFile, gen, dis);
    }

    size_t numRounds = static_cast<size_t>(plantStatesResult.size);
    for (size_t i = 0; i < numRounds; i++) {
        result.push_back(make_unique<RoundScript>(plantStatesResult, i/*1*//*8*//*12*//*205*/, numRounds, gen, dis,
                                                  playerFreeze[i], names[i], cameraPoses[i], cameraAngles[i], 0));
    }
    if (quitAtEnd) {
        result.push_back(make_unique<QuitScript>());
    }
    else {
        result.push_back(make_unique<WaitUntilScoreScript>());
    }

    return result;
}
