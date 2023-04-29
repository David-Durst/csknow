//
// Created by durst on 4/29/23.
//
#include <iostream>
#include <unistd.h>
#include <map>
#include <string>
#include <sstream>
#include <functional>
#include <fstream>
#include <iomanip>
#include <ctime>
#include "load_data.h"
#include "indices/build_indexes.h"
#include "load_cover.h"
#include "load_clusters.h"
#include "queries/velocity.h"
#include "queries/wallers.h"
#include "queries/baiters.h"
#include "queries/netcode.h"
#include "queries/looking.h"
#include "queries/nearest_origin.h"
#include "queries/player_in_cover_edge.h"
#include "queries/team_looking_at_cover_edge_cluster.h"
#include "queries/nonconsecutive.h"
#include "queries/grouping.h"
#include "queries/groupInSequenceOfRegions.h"
#include "queries/base_tables.h"
#include "queries/position_and_wall_view.h"
#include "indices/spotted.h"
#include "queries/nav_visible.h"
#include "queries/nav_danger.h"
#include "queries/nav_cells.h"
#include "queries/distance_to_places.h"
#include "queries/orders.h"
#include "queries/nearest_nav_cell.h"
#include "queries/moments/extract_valid_bot_retakes_rounds.h"
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include <cerrno>
#include "navmesh/nav_file.h"
#include "queries/nav_mesh.h"
#include "queries/reachable.h"
#include "queries/moments/behavior_tree_latent_states.h"
#include <filesystem>
namespace fs = std::filesystem;

using std::map;
using std::string;
using std::reference_wrapper;

int main(int argc, char * argv[]) {
    if (argc != 4) {
        std::cout << "please call this code 3 arguments: " << std::endl;
        std::cout << "1. path/to/local_data" << std::endl;
        std::cout << "2. path/to/nav_meshes" << std::endl;
        std::cout << "3. path/to/output/dir" << std::endl;
        return 1;
    }

    string dataPath = argv[1];
    string navPath = argv[2];
    string outputDir = argv[3];

    std::map<std::string, nav_mesh::nav_file> map_navs;
    //Figure out from where to where you'd like to find a path
    for (const auto & entry : fs::directory_iterator(navPath)) {
        if (entry.path().extension() == ".nav") {
            map_navs.insert(std::pair<std::string, nav_mesh::nav_file>(entry.path().filename().replace_extension(),
                                                                       nav_mesh::nav_file(entry.path().c_str())));
        }
    }

    std::map<std::string, VisPoints> map_visPoints;
    for (const auto & entry : fs::directory_iterator(navPath)) {
        string filename = entry.path().filename();
        // ignore the cells file, will cover both when hit area
        size_t extensionLocation = filename.find(".area.vis.gz");
        if (extensionLocation != string::npos) {
            string mapName = filename.substr(0, filename.find(".area.vis.gz"));
            string extension = filename.substr(filename.find(".area.vis.gz"));
            map_visPoints.insert(std::pair<std::string, VisPoints>(mapName, VisPoints(map_navs[mapName])));
            std::cout << mapName << " num cells: " << map_visPoints.at(mapName).getCellVisPoints().size() << std::endl;
            map_visPoints.at(mapName).load(navPath, mapName, true, map_navs[mapName], true);
            map_visPoints.at(mapName).load(navPath, mapName, false, map_navs[mapName], true);
        }
    }

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream timestampSS;
    timestampSS << std::put_time(&tm, "%d_%m_%Y__%H_%M_%S");
    string timestamp = timestampSS.str();
    std::cout << "timestamp: " << timestamp << std::endl;

    Equipment equipment;
    GameTypes gameTypes;
    HitGroups hitGroups;
    Games games;
    Players players;
    Rounds unfilteredRounds, filteredRounds;
    Ticks ticks;
    PlayerAtTick playerAtTick;
    Spotted spotted;
    Footstep footstep;
    WeaponFire weaponFire;
    Kills kills;
    Hurt hurt;
    Grenades grenades;
    Flashed flashed;
    GrenadeTrajectories grenadeTrajectories;
    Plants plants;
    Defusals defusals;
    Explosions explosions;

    loadData(equipment, gameTypes, hitGroups, games, players, unfilteredRounds, filteredRounds, ticks, playerAtTick, spotted, footstep, weaponFire,
             kills, hurt, grenades, flashed, grenadeTrajectories, plants, defusals, explosions, dataPath);
    buildIndexes(equipment, gameTypes, hitGroups, games, players, filteredRounds, ticks, playerAtTick, spotted, footstep, weaponFire,
                 kills, hurt, grenades, flashed, grenadeTrajectories, plants, defusals, explosions);
    //std::printf("GLIBCXX: %d\n",__GLIBCXX__);
    std::cout << "num elements in equipment: " << equipment.size << std::endl;
    std::cout << "num elements in game_types: " << gameTypes.size << std::endl;
    std::cout << "num elements in hit_groups: " << hitGroups.size << std::endl;
    std::cout << "num elements in games: " << games.size << std::endl;
    std::cout << "num elements in players: " << players.size << std::endl;
    std::cout << "num elements in unfiltered_rounds: " << unfilteredRounds.size << std::endl;
    std::cout << "num elements in filtered_rounds: " << filteredRounds.size << std::endl;
    std::cout << "num elements in ticks: " << ticks.size << std::endl;
    std::cout << "num elements in playerAtTick: " << playerAtTick.size << std::endl;
    std::cout << "num elements in spotted: " << spotted.size << std::endl;
    std::cout << "num elements in footstep: " << footstep.size << std::endl;
    std::cout << "num elements in weaponFire: " << weaponFire.size << std::endl;
    std::cout << "num elements in kills: " << kills.size << std::endl;
    std::cout << "num elements in hurt: " << hurt.size << std::endl;
    std::cout << "num elements in grenades: " << grenades.size << std::endl;
    std::cout << "num elements in flashed: " << flashed.size << std::endl;
    std::cout << "num elements in grenadeTrajectories: " << grenadeTrajectories.size << std::endl;
    std::cout << "num elements in plants: " << plants.size << std::endl;
    std::cout << "num elements in defusals: " << defusals.size << std::endl;
    std::cout << "num elements in explosions: " << explosions.size << std::endl;

    QueryGames queryGames(games);
    QueryRounds queryRounds(games, filteredRounds);
    QueryPlayers queryPlayers(games, players);
    QueryTicks queryTicks(filteredRounds, ticks);
    QueryPlayerAtTick queryPlayerAtTick(filteredRounds, ticks, playerAtTick);


    // core tables
    string dust2MeshName = "de_dust2_mesh";
    MapMeshResult d2MeshResult = queryMapMesh(map_navs["de_dust2"], dust2MeshName);
    string dust2CellsName = "de_dust2_cells";
    MapCellsResult d2CellsResult = queryMapCells(map_visPoints.at("de_dust2"), map_navs["de_dust2"], dust2CellsName);
    string dust2ReachableName = "de_dust2_reachable";
    ReachableResult d2ReachableResult = queryReachable(map_visPoints.at("de_dust2"), d2MeshResult, dust2MeshName, navPath, "de_dust2");
    string dust2DistanceToPlacesName = "de_dust2_distance_to_places";
    DistanceToPlacesResult d2DistanceToPlacesResult = queryDistanceToPlaces(map_navs["de_dust2"], d2ReachableResult,
                                                                            dust2MeshName, navPath, "de_dust2");
    string dust2AreaVisibleName = "de_dust2_area_visible";
    NavVisibleResult d2AreaVisibleResult(dust2MeshName, true, map_visPoints.find("de_dust2")->second, "de_dust2");
    string dust2CellVisibleName = "de_dust2_cell_visible";
    NavVisibleResult d2CellVisibleResult(dust2CellsName, false, map_visPoints.find("de_dust2")->second, "de_dust2");
    string dust2DangerName = "de_dust2_danger";
    NavDangerResult d2NavDangerResult = queryNavDanger(map_visPoints.find("de_dust2")->second, dust2MeshName);

    // orders
    string ordersName = "orders";
    std::cout << "processing orders" << std::endl;
    csknow::orders::OrdersResult ordersResult(map_visPoints.at("de_dust2"), d2MeshResult, d2DistanceToPlacesResult);
    ordersResult.runQuery();
    ordersResult.toCSV(std::cout);
    std::cout << "size: " << ordersResult.size << std::endl;

    // nearest nav areas
    string nearestNavAreasName = "nearestNavAreas";
    std::cout << "processing nearestNavAreas" << std::endl;
    csknow::nearest_nav_cell::NearestNavCell nearestNavCellResult(map_visPoints.at("de_dust2"));
    nearestNavCellResult.runQuery(navPath, "de_dust2");
    std::cout << "size: " << nearestNavCellResult.size << std::endl;

    // plant states
    string plantStatesName = "plantStates";
    std::cout << "processing plantStates" << std::endl;
    csknow::round_extractor::ExtractValidBotRetakesRounds extractValidBotRetakesRounds(games, filteredRounds);

    map<string, reference_wrapper<QueryResult>> analyses {
        //{plantStatesName, plantStatesResult},
    };

    // create the output files and the metadata describing files
    for (const auto & [name, result] : analyses) {
        //std::ofstream fsOverride;
        std::cout << "writing " << outputDir + "/" + name + ".hdf5" << std::endl;
        //fsOverride.open(outputDir + "/" + name + ".csv");
        //result.get().toCSV(fsOverride);
        result.get().toHDF5(outputDir + "/" + name + ".hdf5");
        //fsOverride.close();
    }

    return 0;
}
