//
// Created by durst on 5/29/23.
//

#include "bots/analysis/formation_initializer.h"
#include "file_helpers.h"
#include <filesystem>

namespace csknow::formation_initializer {
    Vec3 FormationInitializer::getValidPlayerCoordinate(const MapMeshResult &mapMeshResult) {
        while (true) {
            int areaIndex = navAreaDist(gen);
            AABB areaAABB = mapMeshResult.coordinate[areaIndex];
            Vec3 areaDelta = areaAABB.max - areaAABB.min;
            if (!mapMeshResult.connectionAreaIds[areaIndex].empty() && areaDelta.x > 2 * WIDTH &&
                areaDelta.y > 2 * WIDTH) {
                Vec3 pos = getCenter(mapMeshResult.coordinate[areaIndex]);
                pos.z += 30;
                return pos;
            }
        }
    }

    FormationInitializer::FormationInitializer(const MapMeshResult & mapMeshResult, const string & savedDataPath) :
        gen(rd()), realDist(0, 1), ctPlayersPerTeamDist(1, static_cast<int>(NUM_PLAYERS / 2) - 1),
        // one fewer for t as fewewer players on that team
        tPlayersPerTeamDist(1, static_cast<int>(NUM_PLAYERS / 2) - 2),
        navAreaDist(0, mapMeshResult.id.size()-1) {
        std::string filePath = savedDataPath + "/de_dust2_formations.hdf5";
        if (std::filesystem::exists(filePath)) {
            load(filePath);
        }
        else {
            for (size_t i = 0; i < numFormations; i++) {
                Formation formation;
                formation.team = i % 2 == 0 ? ENGINE_TEAM_CT : ENGINE_TEAM_T;
                formation.c4PlantedA = realDist(gen) < 0.5;
                int numPlayers = formation.team == ENGINE_TEAM_CT ? ctPlayersPerTeamDist(gen) : tPlayersPerTeamDist(gen);
                for (int playerIndex = 0; playerIndex < numPlayers; playerIndex++) {
                    formation.playerPos.push_back(getValidPlayerCoordinate(mapMeshResult));
                    // for now, everyone on CT is aggresive to prevent lurking deadlocks
                    if (formation.team == ENGINE_TEAM_T) {
                        formation.playerAggressive.push_back(realDist(gen) < 0.5);
                    }
                    else {
                        formation.playerAggressive.push_back(true);
                    }
                }
                initialConditions.push_back(formation);
            }
            save(filePath);
        }
    }

    // We create an empty HDF55 file, by truncating an existing
    // file if required:
    void FormationInitializer::save(const std::string &filePath) {
        HighFive::File file(filePath, HighFive::File::Overwrite);
        vector<double> posX, posY, posZ;
        vector<bool> playerAggressive;
        vector<int> playersPerFormation;
        vector<bool> c4PlantedA;
        vector<TeamId> team;
        for (const auto & formation : initialConditions) {
            for (const auto & playerPos : formation.playerPos) {
                posX.push_back(playerPos.x);
                posY.push_back(playerPos.y);
                posZ.push_back(playerPos.z);
            }
            for (bool aggressive : formation.playerAggressive) {
                playerAggressive.push_back(aggressive);
            }
            playersPerFormation.push_back(formation.playerPos.size());
            c4PlantedA.push_back(formation.c4PlantedA);
            team.push_back(formation.team);
        }
        file.createDataSet("/data/pos x", posX);
        file.createDataSet("/data/pos y", posY);
        file.createDataSet("/data/pos z", posZ);
        file.createDataSet("/data/player aggressive", playerAggressive);
        file.createDataSet("/extra/players per formation", playersPerFormation);
        file.createDataSet("/extra/c4 planted a", c4PlantedA);
        file.createDataSet("/extra/team", team);
    }

    void FormationInitializer::load(const std::string &filePath) {
        HighFive::File file(filePath, HighFive::File::ReadOnly);

        auto posX = file.getDataSet("/data/pos x").read<std::vector<double>>();
        auto posY = file.getDataSet("/data/pos y").read<std::vector<double>>();
        auto posZ = file.getDataSet("/data/pos z").read<std::vector<double>>();
        auto playerAggressive = file.getDataSet("/data/player aggressive").read<std::vector<bool>>();
        auto playersPerFormation = file.getDataSet("/extra/players per formation").read<std::vector<int>>();
        auto c4PlantedA = file.getDataSet("/extra/c4 planted a").read<std::vector<bool>>();
        auto team = file.getDataSet("/extra/team").read<std::vector<TeamId>>();

        int playersInCurFormation = std::numeric_limits<int>::max(), curFormationIndex = -1;
        initialConditions.clear();
        for (size_t i = 0; i < posX.size(); i++) {
            if (curFormationIndex < 0 || playersInCurFormation >= playersPerFormation[curFormationIndex]) {
                playersInCurFormation = 0;
                curFormationIndex++;
                initialConditions.push_back({{}, {}, c4PlantedA[curFormationIndex], team[curFormationIndex]});
            }
            initialConditions.back().playerPos.push_back({posX[i], posY[i], posZ[i]});
            initialConditions.back().playerAggressive.push_back(playerAggressive[i]);
            playersInCurFormation++;
        }
    }

    set<string> findEndOffenseNavPlaces(bool aSite) {
        set<string> result;
        const vector<Order> & offenseOrders = aSite ? strategy::aOffenseOrders : strategy::bOffenseOrders;
        for (const auto & order : offenseOrders) {
            result.insert(order.waypoints.back().placeName);
        }
        return result;
    }

    set<string> findEndDefenseNavPlaces(const MapMeshResult & mapMeshResult, bool aSite) {
        set<string> result;
        const vector<Order> & defenseOrders = aSite ? strategy::aDefenseOrders : strategy::bDefenseOrders;
        for (const auto & order : defenseOrders) {
            for (const auto & waypoint : order.waypoints) {
                if (waypoint.type == WaypointType::HoldPlace) {
                    result.insert(waypoint.placeName);
                }
                else if (waypoint.type == WaypointType::HoldAreas) {
                    for (const auto & area : waypoint.areaIds) {
                        size_t internalAreaId = mapMeshResult.areaToInternalId.at(area);
                        result.insert(mapMeshResult.placeName[internalAreaId]);
                    }
                }
            }
        }
        return result;
    }

    vector<Script::Ptr> FormationInitializer::createFormationScripts(const MapMeshResult & mapMeshResult,
                                                                     bool quitAtEnd) {
        vector<Script::Ptr> result;
        set<string> endAOffenseNavPlaces(findEndOffenseNavPlaces(true)),
            endBOffenseNavPlaces(findEndOffenseNavPlaces(false)),
            endADefenseNavPlaces(findEndDefenseNavPlaces(mapMeshResult, true)),
            endBDefenseNavPlaces(findEndDefenseNavPlaces(mapMeshResult, false));

        for (size_t i = 0; i < initialConditions.size(); i++) {
            const auto & initialCondition = initialConditions[i % 4];
            vector<NeededBot> neededBots;
            for (const auto & aggressive : initialCondition.playerAggressive) {
                neededBots.push_back({0, initialCondition.team,
                                      aggressive ? AggressiveType::Push : AggressiveType::Bait});
            }
            set<string> validStoppingPlaces;
            if (initialCondition.c4PlantedA) {
                if (initialCondition.team == ENGINE_TEAM_CT) {
                    validStoppingPlaces = endAOffenseNavPlaces;
                }
                else {
                    validStoppingPlaces = endADefenseNavPlaces;
                }
            }
            else {
                if (initialCondition.team == ENGINE_TEAM_CT) {
                    validStoppingPlaces = endBOffenseNavPlaces;
                }
                else {
                    validStoppingPlaces = endBDefenseNavPlaces;
                }
            }

            result.push_back(make_unique<FormationScript>("Formation_" + std::to_string(i), neededBots,
                                                          ObserveSettings{ObserveType::FirstPerson, 0}, validStoppingPlaces,
                                                          initialCondition.playerPos, initialCondition.c4PlantedA, initialCondition.team,
                                                          i, initialConditions.size(), false));
        }
        if (quitAtEnd) {
            result.push_back(make_unique<QuitScript>());
        }

        return result;

    }

    FormationScript::FormationScript(const std::string &name, vector<NeededBot> neededBots,
                                     ObserveSettings observeSettings, set<string> validStoppingPlaces,
                                     vector<Vec3> playerPos, bool c4PlantedA, TeamId team,
                                     std::size_t testIndex, std::size_t numTests,
                                     bool waitForever) : Script(name, neededBots, observeSettings),
                                     validStoppingPlaces(validStoppingPlaces), playerPos(playerPos), c4PlantedA(c4PlantedA),
                                     team(team), testIndex(testIndex), numTests(numTests), waitForever(waitForever) { }

    void FormationScript::initialize(Tree &tree, ServerState &state) {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<AreaId> areasToRemove;
            vector<CSGOId> neededBotIds = getNeededBotIds();
            vector<Vec2> playerViewAngle;
            for (size_t i = 0; i < playerPos.size(); i++) {
                playerViewAngle.push_back({0., 0.});
            }
            Vec3 c4Pos = c4PlantedA ? aSiteC4Pos : bSiteC4Pos;
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                        make_unique<InitTestingRound>(blackboard, name),
                                                                        make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                                        make_unique<SlayAllBut>(blackboard, neededBotIds, state),
                                                                        make_unique<TeleportMultiple>(blackboard, neededBotIds, playerPos, playerViewAngle, state),
                                                                        make_unique<SetPos>(blackboard, c4Pos, Vec2({0., 0.})),
                                                                        make_unique<TeleportPlantedC4>(blackboard),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<RecomputeOrdersNode>(blackboard),
                                                                        make_unique<RepeatDecorator>(blackboard,
                                                                                                     make_unique<StandingStill>(blackboard, neededBotIds), true)),
                                                                "Setup");

            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", neededBotIds, false)
            ), "DisableDuringSetup");

            Node::Ptr finishCondition;
            if (waitForever) {
                finishCondition = make_unique<RepeatDecorator>(blackboard,
                                                               make_unique<movement::WaitNode>(blackboard, 30, false),
                                                                       true);
            }
            else {
                vector<Node::Ptr> conditionPlayerNodes;
                for (const auto & neededBot : neededBots) {
                    if (getPlaceAreaModelProbabilities(ENGINE_TEAM_CT) || getPlaceAreaModelProbabilities(ENGINE_TEAM_T)) {
                        conditionPlayerNodes.emplace_back(make_unique<NearPlaces>(blackboard, neededBot.id, validStoppingPlaces, 200));
                    }
                    else {
                        conditionPlayerNodes.emplace_back(make_unique<InPlaces>(blackboard, neededBot.id, validStoppingPlaces));
                    }
                }
                if (team == ENGINE_TEAM_T) {
                    conditionPlayerNodes.emplace_back(make_unique<StandingStill>(blackboard, neededBotIds));
                }
                finishCondition = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                        make_unique<RepeatDecorator>(blackboard, make_unique<ParallelAndNode>(blackboard, std::move(conditionPlayerNodes)), true),
                        make_unique<DisableActionsNode>(blackboard, "disableDefuse", neededBotIds, false, false, false, true),
                        make_unique<csknow::tests::learned::FailIfTimeoutEndNode>(blackboard, name, testIndex, numTests, 70)));
            }

            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<csknow::tests::learned::StartNode>(blackboard, name, testIndex, numTests),
                                                         std::move(finishCondition),
                                                         make_unique<csknow::tests::learned::SuccessEndNode>(blackboard, name, testIndex, numTests)),
                                                 "LearnedNavSequence");
        }
    }

}