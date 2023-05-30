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
            double distanceToSite;
            if (!mapMeshResult.connectionAreaIds[areaIndex].empty()) {
                return getCenter(mapMeshResult.coordinate[areaIndex]);
            }
        }
    }

    FormationInitializer::FormationInitializer(const MapMeshResult & mapMeshResult, const string & navPath) :
        gen(rd()), realDist(0, 1), playersPerTeamDist(0, static_cast<int>(NUM_PLAYERS / 2) - 1),
        navAreaDist(0, mapMeshResult.id.size()-1) {
        std::string filePath = navPath + "/de_dust2_nav_above_below.hdf5";
        if (std::filesystem::exists(filePath)) {
            load(filePath);
        }
        else {
            for (size_t i = 0; i < numFormations; i++) {
                Formation formation;
                formation.team = i % 2 == 0 ? ENGINE_TEAM_CT : ENGINE_TEAM_T;
                formation.c4PlantedA = realDist(gen) < 0.5;
                int numPlayers = playersPerTeamDist(gen);
                for (int playerIndex = 0; playerIndex < numPlayers; playerIndex++) {
                    formation.playerPos.push_back(getValidPlayerCoordinate(mapMeshResult));
                    formation.playerAggressive.push_back(realDist(gen) < 0.5);
                }
                initialConditions.push_back(formation);
            }
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
                curFormationIndex++;
                initialConditions.push_back({{}, {}, c4PlantedA[curFormationIndex], team[curFormationIndex]});
            }
            initialConditions.back().playerPos.push_back({posX[i], posY[i], posZ[i]});
            initialConditions.back().playerAggressive.push_back(playerAggressive[i]);
        }
    }

    vector<Script::Ptr> FormationInitializer::createFormationScripts(bool quitAtEnd) {

    }

    FormationScript::FormationScript(const std::string &name, vector<NeededBot> neededBots,
                                     ObserveSettings observeSettings, std::size_t testIndex, std::size_t numTests,
                                     bool waitForever) : Script(name, neededBots, observeSettings),
                                     testIndex(testIndex), numTests(numTests), waitForever(waitForever) { }

    void FormationScript::initialize(Tree &tree, ServerState &state, vector<Vec3> playerPos, bool c4PlantedA) {
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
                                                                        make_unique<RepeatDecorator>(blackboard,
                                                                                                     make_unique<StandingStill>(blackboard, neededBotIds), true)),
                                                                "Setup");

            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", neededBotIds, false)
            ), "DisableDuringSetup");

            Node::Ptr finishCondition;
            /*
            if (waitForever) {
                finishCondition = make_unique<RepeatDecorator>(blackboard,
                                                               make_unique<movement::WaitNode>(blackboard, 30, false),
                                                                       true);
            }
            else {
                Node::Ptr pusherSeesEnemyBeforeLurkerMoves =
                        make_unique<RepeatDecorator>(blackboard, make_unique<ParallelAndNode>(blackboard, Node::makeList(
                                make_unique<InPlace>(blackboard, neededBots[0].id, "LongDoors"),
                                make_unique<InPlace>(blackboard, neededBots[1].id, "OutsideLong"),
                                make_unique<InPlace>(blackboard, neededBots[2].id, "ExtendedA")
                        )), true);

                finishCondition = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                        make_unique<RepeatDecorator>(blackboard, std::move(condition), true),
                        make_unique<csknow::tests::learned::FailIfTimeoutEndNode>(blackboard, name, testIndex, numTests, 30)));
            }

            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<StartNode>(blackboard, name, testIndex, numTests),
                                                         std::move(finishCondition),
                                                         make_unique<SuccessEndNode>(blackboard, name, testIndex, numTests)),
                                                 "LearnedNavSequence");
                                                 */
        }
    }

}