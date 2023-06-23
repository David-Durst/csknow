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
#include "queries/base_tables.h"
#include "queries/nav_visible.h"
#include "queries/nav_danger.h"
#include "queries/nav_cells.h"
#include "queries/distance_to_places.h"
#include "queries/orders.h"
#include "queries/nearest_nav_cell.h"
#include "queries/moments/aggression_event.h"
#include "queries/moments/fire_history.h"
#include "queries/moments/engagement.h"
#include "queries/moments/engagement_per_tick_aim.h"
#include "queries/moments/non_engagement_trajectory.h"
#include "queries/moments/trajectory_segments.h"
#include "queries/moments/latent_extractors/latent_engagement.h"
#include "queries/training_moments/training_engagement_aim.h"
#include "queries/inference_moments/inference_engagement_aim.h"
#include "queries/training_moments/training_navigation.h"
#include "queries/moments/key_retake_events.h"
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include <cerrno>
#include "navmesh/nav_file.h"
#include "queries/nav_mesh.h"
#include "queries/reachable.h"
#include "queries/moments/behavior_tree_latent_states.h"
#include "bots/analysis/nav_area_above_below.h"
#include <filesystem>
namespace fs = std::filesystem;

using std::map;
using std::string;
using std::reference_wrapper;

int main(int argc, char * argv[]) {
    if (argc != 4 && argc != 5 && argc != 6) {
        std::cout << "please call this code with 3 or 4 or 5 arguments: " << std::endl;
        std::cout << "1. path/to/local_data" << std::endl;
        std::cout << "2. path/to/nav_meshes" << std::endl;
        std::cout << "3. path/to/output/dir" << std::endl;
        std::cout << "4. y to enable non-test plant rounds" << std::endl;
        std::cout << "5. appendix for output name" << std::endl;
        return 1;
    }

    string dataPath = argv[1];
    string navPath = argv[2];
    string outputDir = argv[3];
    bool enableNonTestPlantRounds = argc >= 5;
    string outputNameAppendix = "";
    if (argc == 6) {
        outputNameAppendix = argv[5];
    }

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
    //csknow::navigation::testNavImages(map_visPoints.at("de_dust2"), outputDir);

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
    Rounds unfilteredRounds(false), filteredRounds(true);
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
    Say say;

    loadDataHDF5(equipment, gameTypes, hitGroups, games, players, unfilteredRounds, filteredRounds, ticks, playerAtTick,
                spotted, footstep, weaponFire,
                kills, hurt, grenades, flashed, grenadeTrajectories, plants, defusals, explosions, say, dataPath);
    buildIndexes(equipment, gameTypes, hitGroups, games, players, filteredRounds, ticks, playerAtTick, spotted, footstep, weaponFire,
                 kills, hurt, grenades, flashed, grenadeTrajectories, plants, defusals, explosions, say);
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
    std::cout << "num elements in say: " << say.size << std::endl;

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

    std::cout << "places: ";
    for (size_t i = 0; i < d2DistanceToPlacesResult.places.size(); i++) {{
        std::cout << "(" << i << "," << d2DistanceToPlacesResult.places[i] << "); ";
    }}
    std::cout << std::endl;

    csknow::nav_area_above_below::NavAreaAboveBelow d2AboveBelow(d2MeshResult, navPath);


    csknow::key_retake_events::KeyRetakeEvents keyRetakeEvents(filteredRounds, ticks, playerAtTick, plants, defusals,
                                                               kills, say);

    /*
    std::cout << "closest distance " << d2DistanceToPlacesResult.getClosestDistance(1742, "ExtendedA", map_navs.at("de_dust2")) << std::endl;
    std::cout << "short area ids: ";
    for (const auto & shortAreaId : d2DistanceToPlacesResult.placeToArea["Short"]) {
        std::cout << shortAreaId << ", ";
    }
    std::cout << std::endl;
    std::cout << "closest distance " << d2DistanceToPlacesResult.getClosestDistance("Short", "ExtendedA", map_navs.at("de_dust2")) << std::endl;
    std::cout << "distance " << d2ReachableResult.getDistance(1742, 1728, map_navs.at("de_dust2"));
    exit(0);
    for (const auto & tickIndex : {1476979, 1477226, 1477162, 1477098, 1477034}) {
        int64_t roundIndex = ticks.roundId[tickIndex];
        int64_t gameIndex = filteredRounds.gameId[roundIndex];
        std::cout << "tick index: " << tickIndex << " game tick index " << ticks.gameTickNumber[tickIndex]
            << " demo file " << games.demoFile[gameIndex] << std::endl;

    }
    int64_t tmpTickIndex = 1476979;
    int64_t tmpRoundIndex = ticks.roundId[tmpTickIndex];
    int64_t tmpGameIndex = filteredRounds.gameId[tmpRoundIndex];
    for (int64_t tmpR = games.roundsPerGame[tmpGameIndex].minId; tmpR <= games.roundsPerGame[tmpGameIndex].maxId; tmpR++) {
        std::cout << "tmpR " << tmpR << " start index " << filteredRounds.ticksPerRound[tmpR].minId
            << " start game tick number " << ticks.gameTickNumber[filteredRounds.ticksPerRound[tmpR].minId]
            << " end tick number " << filteredRounds.ticksPerRound[tmpR].maxId
            << " end game tick number " << ticks.gameTickNumber[filteredRounds.ticksPerRound[tmpR].maxId]
            << " real end tick number " << filteredRounds.endTick[tmpR]
            << " real game end tick number " << ticks.gameTickNumber[filteredRounds.endTick[tmpR]] << std::endl;
    }
    exit(0);
     */


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

    /*
    // fire history
    string fireHistoryName = "fireHistory";
    std::cout << "processing fire history" << std::endl;
    csknow::fire_history::FireHistoryResult fireHistoryResult(filteredRounds, ticks);
    fireHistoryResult.runQuery(games, weaponFire, hurt, playerAtTick);
    std::cout << "size: " << fireHistoryResult.size << std::endl;
     */

    // engagement events
    string engagementName = "engagement";
    std::cout << "processing engagements" << std::endl;
    EngagementResult engagementResult = queryEngagementResult(games, filteredRounds, ticks, hurt);
    std::cout << "size: " << engagementResult.size << std::endl;

    // bt latent events
    string behaviorTreeLatentEventsName = "behaviorTreeLatentEvents";
    string behaviorTreeFeatureStoreName = "behaviorTreeFeatureStore";
    string behaviorTreeTeamFeatureStoreName = "behaviorTreeTeamFeatureStore";
    string behaviorTreeWindowFeatureStoreName = "behaviorTreeWindowFeatureStore";
    std::cout << "processing behaviorTreeLatentEvents" << std::endl;
    keyRetakeEvents.enableNonTestPlantRounds = enableNonTestPlantRounds;
    csknow::behavior_tree_latent_states::BehaviorTreeLatentStates behaviorTreeLatentEvents(ticks, playerAtTick,
                                                                                           ordersResult.orders,
                                                                                           keyRetakeEvents);
    behaviorTreeLatentEvents.runQuery(navPath + "/de_dust2.nav", map_visPoints.at("de_dust2"), d2MeshResult,
                                      d2ReachableResult, d2DistanceToPlacesResult,
                                      nearestNavCellResult, d2AboveBelow,
                                      ordersResult, players, games, filteredRounds, ticks,
                                      playerAtTick, weaponFire, hurt, plants, defusals, engagementResult);
    std::cout << "size: " << behaviorTreeLatentEvents.size << std::endl;
    //behaviorTreeLatentEvents.featureStoreResult.teamFeatureStoreResult.checkPossiblyBadValue();

    std::cout << "processing behavior tree feature store" << std::endl;
    //behaviorTreeLatentEvents.featureStoreResult.computeAcausalLabels(games, filteredRounds, ticks, playerAtTick);
    std::cout << "size: " << behaviorTreeLatentEvents.featureStoreResult.size << std::endl;

    std::cout << "processing behavior tree team feature store" << std::endl;
    behaviorTreeLatentEvents.featureStoreResult.teamFeatureStoreResult.computeAcausalLabels(games, filteredRounds, ticks,
                                                                                            players, d2DistanceToPlacesResult,
                                                                                            map_navs.at("de_dust2"),
                                                                                            keyRetakeEvents);
    std::cout << "size: " << behaviorTreeLatentEvents.featureStoreResult.teamFeatureStoreResult.size << std::endl;

    /*
    std::cout << "processing behavior tree window feature store" << std::endl;
    csknow::feature_store::FeatureStoreResult windowFeatureStoreResult =
        behaviorTreeLatentEvents.featureStoreResult.makeWindows();
    std::cout << "size: " << windowFeatureStoreResult.size << std::endl;
     */

    /*
    // latent engagement events
    string latentEngagementName = "latentEngagement";
    std::cout << "processing latent engagements" << std::endl;
    csknow::latent_engagement::LatentEngagementResult latentEngagementResult;
    latentEngagementResult.runQuery(filteredRounds, ticks, hurt, behaviorTreeLatentEvents);
    std::cout << "size: " << latentEngagementResult.size << std::endl;

    // non-engagement trajectories
    std::cout << "processing non engagement trajectory" << std::endl;
    string nonEngagementTrajectoryName = "nonEngagementTrajectory";
    NonEngagementTrajectoryResult nonEngagementTrajectoryResult =
            queryNonEngagementTrajectory(filteredRounds, ticks, playerAtTick, engagementResult);
    std::cout << "size: " << nonEngagementTrajectoryResult.size << std::endl;
    std::cout << "processing trajectory segments" << std::endl;

    // trajectory data
    string trajectorySegmentName = "trajectorySegment";
    TrajectorySegmentResult trajectorySegmentResult =
            queryAllTrajectories(players, games, filteredRounds, ticks, playerAtTick, nonEngagementTrajectoryResult);
    std::cout << "size: " << trajectorySegmentResult.size << std::endl;
    std::cout << "processing training navigation data set" << std::endl;
    string trainingNavigationName = "trainNav";
    / *
    csknow::navigation::TrainingNavigationResult trainingNavigationResult; =
        csknow::navigation::queryTrainingNavigation(map_visPoints.at("de_dust2"), d2ReachableResult, players, games,
                                                    filteredRounds, ticks, playerAtTick, nonEngagementTrajectoryResult,
                                                    outputDir, true);
    std::cout << "size: " << trainingNavigationResult.size << std::endl;


    // engagement aim data
    std::cout << "processing training engagement aim training data set" << std::endl;
    string engagementAimName = "engagementAim";
    TrainingEngagementAimResult engagementAimResult =
        queryTrainingEngagementAim(games, filteredRounds, ticks, playerAtTick, engagementResult,
                                   fireHistoryResult, map_visPoints.at("de_dust2"), nearestNavCellResult);
    std::cout << "size: " << engagementAimResult.size << std::endl;

    // latent engagement aim data
    / *
    std::cout << "processing training latent engagement aim training data set" << std::endl;
    string latentEngagementAimName = "latentEngagementAim";
    TrainingEngagementAimResult latentEngagementAimResult =
        queryTrainingEngagementAim(games, filteredRounds, ticks, playerAtTick, latentEngagementResult,
                                   fireHistoryResult, map_visPoints.at("de_dust2"), nearestNavCellResult);
    std::cout << "size: " << latentEngagementAimResult.size << std::endl;
     */

    map<string, reference_wrapper<QueryResult>> analyses {
            //{engagementAimName, engagementAimResult},
            //{latentEngagementAimName, latentEngagementAimResult},
            //{behaviorTreeFeatureStoreName, behaviorTreeLatentEvents.featureStoreResult},
            {behaviorTreeTeamFeatureStoreName + outputNameAppendix, behaviorTreeLatentEvents.featureStoreResult.teamFeatureStoreResult}
            //{trainingNavigationName, trainingNavigationResult},
    };

    auto writeStart = std::chrono::system_clock::now();
    // create the output files and the metadata describing files
    for (const auto & [name, result] : analyses) {
        //std::ofstream fsOverride;
        std::cout << "writing " << outputDir + "/" + name + ".hdf5" << std::endl;
        //fsOverride.open(outputDir + "/" + name + ".csv");
        //result.get().toCSV(fsOverride);
        result.get().toHDF5(outputDir + "/" + name + ".hdf5");
        //fsOverride.close();
    }
    auto writeEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> writeTime = writeEnd - writeStart;
    std::cout << "write time " << writeTime.count() << std::endl;

    return 0;
}
