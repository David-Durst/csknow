//
// Created by durst on 5/29/23.
//

#ifndef CSKNOW_FORMATION_INITIALIZER_H
#define CSKNOW_FORMATION_INITIALIZER_H

#include "bots/load_save_bot_data.h"
#include "queries/nav_mesh.h"
#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"
#include "bots/testing/scripts/learned/log_nodes.h"

namespace csknow::formation_initializer {
    size_t numFormations = 2000;
    Vec3 aSiteC4Pos{1241., 2586., 127.}, bSiteC4Pos{-1463., 2489., 46.};
    struct Formation {
        vector<Vec3> playerPos;
        vector<bool> playerAggressive;
        bool c4PlantedA;
        TeamId team;
    };

    class FormationInitializer {
        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<> realDist;
        std::uniform_int_distribution<> playersPerTeamDist, navAreaDist;

        Vec3 getValidPlayerCoordinate(const MapMeshResult & mapMeshResult);
    public:
        vector<Formation> initialConditions;

        FormationInitializer(const MapMeshResult & mapMeshResult, const string & navPath);

        void save(const string& filePath);
        void load(const string& filePath);

        vector<Script::Ptr> createFormationScripts(const MapMeshResult & mapMeshResult, bool quitAtEnd);
    };

    class FormationScript : public Script {
    public:
        OrderId addedOrderId;
        set<string> validStoppingPlaces;
        vector<Vec3> playerPos;
        bool c4PlantedA;
        size_t testIndex, numTests;
        bool waitForever;
        FormationScript(const std::string &name, vector<NeededBot> neededBots, ObserveSettings observeSettings,
                        set<string> validStoppingPlaces, vector<Vec3> playerPos, bool c4PlantedA,
                        size_t testIndex, size_t numTests, bool waitForever);

        void initialize(Tree &tree, ServerState &state);
    };
}

#endif //CSKNOW_FORMATION_INITIALIZER_H
