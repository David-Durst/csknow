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
#include "queries/grenade/smoke_grenade.h"
#include "queries/grenade/player_flashed.h"
#include "queries/position_and_wall_view.h"
#include "indices/spotted.h"
#include "queries/nav_visible.h"
#include "queries/nav_danger.h"
#include "queries/nav_cells.h"
#include "queries/distance_to_places.h"
#include "queries/moments/aggression_event.h"
#include "queries/moments/engagement.h"
#include "queries/moments/engagement_per_tick_aim.h"
#include "queries/moments/non_engagement_trajectory.h"
#include "queries/moments/trajectory_segments.h"
#include "queries/training_moments/training_engagement_aim.h"
#include "queries/inference_moments/inference_engagement_aim.h"
#include "queries/inference_moments/inference_latent_engagement.h"
#include "queries/inference_moments/inference_latent_engagement_distribution.h"
#include "queries/inference_moments/inference_latent_aggression.h"
#include "queries/inference_moments/inference_latent_aggression_distribution.h"
#include "queries/inference_moments/inference_latent_order.h"
#include "queries/inference_moments/inference_latent_order_distribution.h"
#include "queries/inference_moments/inference_latent_place.h"
#include "queries/inference_moments/inference_latent_place_distribution.h"
#include "queries/training_moments/training_navigation.h"
#include "queries/moments/trajectory_segments.h"
#include "queries/moments/latent_extractors/latent_engagement.h"
#include "queries/moments/latent_extractors/engagement_ticks_until_hurt_kill.h"
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include <cerrno>
#include "navmesh/nav_file.h"
#include "queries/nav_mesh.h"
#include "queries/reachable.h"
#include "queries/orders.h"
#include <filesystem>
namespace fs = std::filesystem;

using std::map;
using std::string;
using std::reference_wrapper;

void exec(const string & cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::string cmdWithPipe = cmd + " 2>&1";
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmdWithPipe.c_str(), "r"), pclose);
    if (!pipe) {
        std::cerr << "error code: " << strerror(errno) << std::endl;
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    std::cout << result;
}

int main(int argc, char * argv[]) {
    if (argc != 5 && argc != 6) {
        std::cout << "please call this code 5 arguments: " << std::endl;
        std::cout << "1. path/to/local_data" << std::endl;
        std::cout << "2. path/to/nav_meshes" << std::endl;
        std::cout << "3. run server (y or n)" << std::endl;
        std::cout << "4. path/to/output/dir" << std::endl;
        std::cout << "5. path/to/models/dir (optional)" << std::endl;
        return 1;
    }

    string dataPath = argv[1];
    string navPath = argv[2];
    bool runServer = argv[3][0] == 'y';
    string outputDir = argv[4];
    bool haveModels = false;
    string modelsDir = "";
    if (argc == 6) {
        haveModels = true;
        modelsDir = argv[5];
    }
    (void) haveModels;

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
    csknow::navigation::testNavImages(map_visPoints.at("de_dust2"), outputDir);

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
    CoverEdges coverEdges;
    CoverOrigins coverOrigins;

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

    //loadCover(coverOrigins, coverEdges, dataPath);
    //buildCoverIndex(coverOrigins, coverEdges);

    //std::cout << "num elements in cover origins: " << coverOrigins.size << std::endl;
    //std::cout << "num elements in cover edges: " << coverEdges.size << std::endl;

    QueryGames queryGames(games);
    QueryRounds queryRounds(games, filteredRounds);
    QueryPlayers queryPlayers(games, players);
    QueryTicks queryTicks(filteredRounds, ticks);
    QueryPlayerAtTick queryPlayerAtTick(filteredRounds, ticks, playerAtTick);
    QueryWeaponFire queryWeaponFire(filteredRounds, ticks, weaponFire);
    QueryHurt queryHurt(filteredRounds, ticks, hurt);
    QueryKills queryKills(filteredRounds, ticks, kills);
    csknow::smoke_grenade::SmokeGrenadeResult smokeGrenadeResult;
    smokeGrenadeResult.runQuery(filteredRounds, ticks, grenades, grenadeTrajectories);
    csknow::player_flashed::PlayerFlashedResult playerFlashedResult;
    playerFlashedResult.runQuery(games, filteredRounds, ticks, playerAtTick, flashed);

    /*
    // record locations and view angles
    std::ofstream fsACatPeekers, fsMidCTPeekers;
    PositionsAndWallViews aCatPeekers = queryViewsFromRegion(rounds, ticks, playerAtTick,
                                                   dataPath + "/../analytics/walls/aCatStanding.csv",
                                                   dataPath + "/../analytics/walls/aCatWalls.csv");
    string aCatPeekersName = "a_cat_peekers";
    fsACatPeekers.open(outputDir + "/" + aCatPeekersName + ".csv" );
    fsACatPeekers << aCatPeekers.toCSV();
    fsACatPeekers.flush();
    fsACatPeekers.close();

    PositionsAndWallViews midCTPeekers = queryViewsFromRegion(rounds, ticks, playerAtTick,
                                                             dataPath + "/../analytics/walls/midCTStanding.csv",
                                                             dataPath + "/../analytics/walls/midWalls.csv");
    string midCTPeekersName = "mid_ct_peekers";
    fsMidCTPeekers.open(outputDir + "/" + midCTPeekersName + ".csv");
    fsMidCTPeekers << midCTPeekers.toCSV();
    fsMidCTPeekers.flush();
    fsMidCTPeekers.close();

    string runClustersPythonCmd("bash " + dataPath + "/../python_analytics/makeClusters.sh");
    exec(runClustersPythonCmd);
    /  *
    int clustersCmdResult = std::system(runClustersPythonCmd.c_str());
    if (clustersCmdResult != 0) {
        std::cout << "clusters cmd result: " << clustersCmdResult << std::endl;
    }
     * /

    // import clusters, track cluster sequences
    std::ofstream fsACatSequences, fsMidCTSequences;
    Cluster aCatPeekersClusters(dataPath + "/../python_analytics/csknow_python_analytics/a_cat_peekers_clusters.csv");
    ClusterSequencesByRound aCatClusterSequence = analyzeViewClusters(rounds, players, playerAtTick, aCatPeekers,
                                                                      aCatPeekersClusters);

    string aCatSequenceName = "a_cat_cluster_sequence";
    fsACatSequences.open(outputDir + "/" + aCatSequenceName + ".csv");
    fsACatSequences << aCatClusterSequence.toCSV();
    fsACatSequences.flush();
    fsACatSequences.close();

    Cluster midCTPeekersClusters(dataPath + "/../python_analytics/csknow_python_analytics/mid_ct_peekers_clusters.csv");
    ClusterSequencesByRound midCTClusterSequence = analyzeViewClusters(rounds, players, playerAtTick, midCTPeekers,
                                                                       midCTPeekersClusters);

    string midCTSequenceName = "mid_ct_cluster_sequence";
    fsMidCTSequences.open(outputDir + "/" + midCTSequenceName + ".csv");
    fsMidCTSequences << midCTClusterSequence.toCSV();
    fsMidCTSequences.flush();
    fsMidCTSequences.close();

    string runTMPythonCmd(dataPath + "/../python_analytics/makeTransitionMatrices.sh");
    int tmCmdResult = std::system(runTMPythonCmd.c_str());
    if (tmCmdResult != 0) {
        std::cout << "transition matrices cmd result: " << tmCmdResult << std::endl;
    }
    */
    /*
    SpottedIndex spottedIndex(position, spotted);
    std::cout << "built spotted index" << std::endl;

    string lookerName = "lookers";
    LookingResult lookersResult = queryLookers(games, filteredRounds, ticks, playerAtTick);
    std::cout << "looker entries: " << lookersResult.tickId.size() << std::endl;

    string nearestOriginName = "nearest_origin";
    NearestOriginResult nearestOriginResult = queryNearestOrigin(rounds, ticks, playerAtTick, coverOrigins);
    std::cout << "nearest_origin entries: " << nearestOriginResult.tickId.size() << std::endl;

    string playerInCoverEdgeName = "player_in_cover_edge";
    PlayerInCoverEdgeResult playerInCoverEdgeResult = queryPlayerInCoverEdge(rounds, ticks, playerAtTick, coverOrigins,
                                                                             coverEdges, nearestOriginResult);
    std::cout << "player_in_cover_edge entries: " << playerInCoverEdgeResult.tickId.size() << std::endl;

    string teamLookingAtCoverEdgeClusterName = "team_looking_at_cover_edge_cluster";
    TeamLookingAtCoverEdgeCluster teamLookingAtCoverEdgeClusterResult =
            queryTeamLookingAtCoverEdgeCluster(games, rounds, ticks, playerAtTick, coverOrigins, coverEdges,
                                               nearestOriginResult);
    std::cout << "team_looking_at_cover_edge_cluster entries: " << teamLookingAtCoverEdgeClusterResult.tickId.size() << std::endl;
    */

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

    // engagements
    std::cout << "processing engagements" << std::endl;
    string engagementName = "engagement";
    EngagementResult engagementResult = queryEngagementResult(games, filteredRounds, ticks, hurt);
    std::cout << "size: " << engagementResult.size << std::endl;

    // bt latent events
    string behaviorTreeLatentEventsName = "behaviorTreeLatentEvents";
    std::cout << "processing behaviorTreeLatentEvents" << std::endl;
    csknow::behavior_tree_latent_states::BehaviorTreeLatentStates behaviorTreeLatentEvents(ticks, playerAtTick, ordersResult.orders);
    behaviorTreeLatentEvents.runQuery(navPath + "/de_dust2.nav", map_visPoints.at("de_dust2"), d2MeshResult,
                                      d2ReachableResult, d2DistanceToPlacesResult,
                                      nearestNavCellResult, ordersResult, players, games, filteredRounds, ticks,
                                      playerAtTick, weaponFire, hurt, plants, defusals, engagementResult);
    std::cout << "size: " << behaviorTreeLatentEvents.size << std::endl;
    behaviorTreeLatentEvents.featureStoreResult.computeAcausalLabels(games, filteredRounds, ticks, playerAtTick);
    behaviorTreeLatentEvents.featureStoreResult.checkInvalid();

    // latent engagement events
    string latentEngagementName = "latentEngagement";
    std::cout << "processing latent engagements" << std::endl;
    csknow::latent_engagement::LatentEngagementResult latentEngagementResult;
    latentEngagementResult.runQuery(filteredRounds, ticks, hurt, behaviorTreeLatentEvents);
    latentEngagementResult.computePercentMatchNearestCrosshair(filteredRounds, ticks, playerAtTick,
                                                               behaviorTreeLatentEvents.featureStoreResult);
    std::cout << "size: " << latentEngagementResult.size << std::endl;

    string engagmeentTicksUntilHurtKill = "engagementTicksUntilHurtKill";
    std::cout << "processing engagement ticks until hurt kill" << std::endl;
    csknow::latent_engagement::EngagementTicksUntilHurtKill engagementTicksUntilHurtKill;
    engagementTicksUntilHurtKill.runQuery(filteredRounds, ticks, playerAtTick, hurt, kills, behaviorTreeLatentEvents);
    std::cout << "size: " << engagementTicksUntilHurtKill.size << std::endl;

    // inference latent engagement
    // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
    // want high intra-op parallelism but not inter-op parallelism
    // intra-op
    //at::set_num_threads(1);
    // inter-op
    at::set_num_interop_threads(1);

    string inferenceLatentOrderName = "inferenceLatentOrder";
    std::cout << "processing inference latent orders" << std::endl;
    csknow::inference_latent_order::InferenceLatentOrderResult inferenceLatentOrderResult;
    inferenceLatentOrderResult.runQuery(modelsDir, filteredRounds, ticks, playerAtTick, behaviorTreeLatentEvents);
    std::cout << "size: " << inferenceLatentOrderResult.size << std::endl;

    string inferenceLatentOrderDistributionName = "inferenceLatentOrderDistribution";
    csknow::inference_latent_order::InferenceLatentOrderDistributionResult
        inferenceLatentOrderDistributionResult(playerAtTick, queryPlayerAtTick, ordersResult, d2MeshResult,
                                               inferenceLatentOrderResult);
    inferenceLatentOrderResult.perTickPosLabelsQuery = inferenceLatentOrderDistributionName;

    string inferenceLatentPlaceName = "inferenceLatentPlace";
    std::cout << "processing inference latent orders" << std::endl;
    csknow::inference_latent_place::InferenceLatentPlaceResult inferenceLatentPlaceResult(d2DistanceToPlacesResult);
    inferenceLatentPlaceResult.runQuery(modelsDir, filteredRounds, ticks, playerAtTick, behaviorTreeLatentEvents);
    std::cout << "size: " << inferenceLatentPlaceResult.size << std::endl;

    string inferenceLatentPlaceDistributionName = "inferenceLatentPlaceDistribution";
    csknow::inference_latent_place::InferenceLatentPlaceDistributionResult
            inferenceLatentPlaceDistributionResult(playerAtTick, queryPlayerAtTick, d2DistanceToPlacesResult,
                                                   inferenceLatentPlaceResult);
    inferenceLatentPlaceResult.perTickPosLabelsQuery = inferenceLatentPlaceDistributionName;

    string inferenceLatentEngagementName = "inferenceLatentEngagement";
    std::cout << "processing inference latent engagements" << std::endl;
    csknow::inference_latent_engagement::InferenceLatentEngagementResult inferenceLatentEngagementResult(playerAtTick);
    inferenceLatentEngagementResult.runQuery(modelsDir, filteredRounds, ticks, behaviorTreeLatentEvents);
    inferenceLatentEngagementResult.computePercentMatchNearestCrosshair(filteredRounds, ticks, playerAtTick,
                                                                        behaviorTreeLatentEvents.featureStoreResult);
    std::cout << "size: " << inferenceLatentEngagementResult.size << std::endl;

    string inferenceLatentEngagementDistributionName = "inferenceLatentEngagementDistribution";
    csknow::inference_latent_engagement::InferenceLatentEngagementDistributionResult
        inferenceLatentEngagementDistributionResult(playerAtTick, queryPlayerAtTick, inferenceLatentEngagementResult);
    inferenceLatentEngagementResult.perTickPlayerLabelsQuery = inferenceLatentEngagementDistributionName;

    // inference latent aggression
    string inferenceLatentAggressionName = "inferenceLatentAggression";
    std::cout << "processing inference latent aggression" << std::endl;
    csknow::inference_latent_aggression::InferenceLatentAggressionResult inferenceLatentAggressionResult(playerAtTick);
    inferenceLatentAggressionResult.runQuery(modelsDir, filteredRounds, ticks, behaviorTreeLatentEvents);
    std::cout << "size: " << inferenceLatentAggressionResult.size << std::endl;

    string inferenceLatentAggressionDistributionName = "inferenceLatentAggressionDistribution";
    csknow::inference_latent_aggression::InferenceLatentAggressionDistributionResult
        inferenceLatentAggressionDistributionResult(playerAtTick, queryPlayerAtTick, inferenceLatentAggressionResult);
    inferenceLatentAggressionResult.perTickPlayerLabelsQuery = inferenceLatentAggressionDistributionName;
    /*
    string inferenceLatentEngagementHitName = "inferenceLatentEngagementHit";
    std::cout << "processing inference latent engagements hit" << std::endl;
    csknow::inference_latent_engagement::InferenceLatentEngagementResult inferenceLatentEngagementHitResult(playerAtTick);
    inferenceLatentEngagementHitResult.runQuery(modelsDir, filteredRounds, ticks, behaviorTreeLatentEvents, true);
    std::cout << "size: " << inferenceLatentEngagementHitResult.size << std::endl;

    string inferenceLatentEngagementVisibleName = "inferenceLatentEngagementVisible";
    std::cout << "processing inference latent engagements visible" << std::endl;
    csknow::inference_latent_engagement::InferenceLatentEngagementResult inferenceLatentEngagementVisibleResult(playerAtTick);
    inferenceLatentEngagementVisibleResult.runQuery(modelsDir, filteredRounds, ticks, behaviorTreeLatentEvents, false);
    std::cout << "size: " << inferenceLatentEngagementVisibleResult.size << std::endl;
     */

    /*
    std::cout << "processing aggression_event" << std::endl;
    string aggressionEventName = "aggression_event";
    AggressionEventResult aggressionEventResult =
            queryAggressionRoles(games, filteredRounds, ticks, playerAtTick, map_navs["de_dust2"], map_visPoints.find("de_dust2")->second, d2ReachableResult);
            */
    std::cout << "processing non engagement trajectory" << std::endl;
    string nonEngagementTrajectoryName = "nonEngagementTrajectory";
    NonEngagementTrajectoryResult nonEngagementTrajectoryResult =
            queryNonEngagementTrajectory(filteredRounds, ticks, playerAtTick, engagementResult);
    std::cout << "size: " << nonEngagementTrajectoryResult.size << std::endl;
    std::cout << "processing trajectory segments" << std::endl;
    string trajectorySegmentName = "trajectorySegment";
    TrajectorySegmentResult trajectorySegmentResult =
            queryAllTrajectories(players, games, filteredRounds, ticks, playerAtTick, nonEngagementTrajectoryResult);
    std::cout << "size: " << trajectorySegmentResult.size << std::endl;

    /*
     disabling aim model stuff
    string fireHistoryName = "fireHistory";
    std::cout << "processing fire history" << std::endl;
    csknow::fire_history::FireHistoryResult fireHistoryResult(filteredRounds, ticks);
    fireHistoryResult.runQuery(games, weaponFire, hurt, playerAtTick);
    std::cout << "size: " << fireHistoryResult.size << std::endl;
    std::cout << "processing training engagement aim training data set" << std::endl;
    string engagementAimName = "trainEngagementAim";
    TrainingEngagementAimResult engagementAimResult =
        queryTrainingEngagementAim(games, filteredRounds, ticks, playerAtTick, engagementResult, fireHistoryResult,
                                   map_visPoints.at("de_dust2"), nearestNavCellResult);
    std::cout << "size: " << engagementAimResult.size << std::endl;
    std::cout << "processing engagements per tick aim" << std::endl;
    string engagementPerTickAimName = "engagementPerTickAim";
    EngagementPerTickAimResult engagementPerTickAimResult =
        queryEngagementPerTickAim(games, filteredRounds, ticks, engagementResult, engagementAimResult);
    engagementResult.havePerTickAimTable = true;
    engagementResult.perTickAimTable = engagementPerTickAimName;
    std::cout << "size: " << engagementPerTickAimResult.size << std::endl;
    std::cout << "processing training navigation data set" << std::endl;
    string trainingNavigationName = "trainNav";
    csknow::navigation::TrainingNavigationResult trainingNavigationResult =
        csknow::navigation::queryTrainingNavigation(map_visPoints.at("de_dust2"), d2ReachableResult, players, games,
                                                    filteredRounds, ticks, playerAtTick, nonEngagementTrajectoryResult,
                                                    outputDir, false);
    std::cout << "size: " << trainingNavigationResult.size << std::endl;
    std::cout << "processing inference engagement aim training data set" << std::endl;
    string inferenceEngagementAimName = "engagementAim";
    InferenceEngagementAimResult inferenceEngagementAimResult(engagementAimResult);
    if (haveModels) {
        inferenceEngagementAimResult.runQuery(filteredRounds, modelsDir, engagementResult);
    }
    std::cout << "size: " << inferenceEngagementAimResult.size << std::endl;
    engagementResult.havePerTickAimPredictionTable = true;
    engagementResult.perTickPredictionAimTable = inferenceEngagementAimName;
     */

    /*
    VelocityResult velocityResult = queryVelocity(position);
    std::cout << "velocity moments: " << velocityResult.positionIndex.size() << std::endl;
    WallersResult wallersResult = queryWallers(position, spotted);
    std::cout << "waller moments: " << wallersResult.positionIndex.size() << std::endl;
    BaitersResult baitersResult = queryBaiters(position, kills, spottedIndex);
    std::cout << "baiter moments: " << baitersResult.positionIndex.size() << std::endl;
    NetcodeResult netcodeResult = queryNetcode(position, weaponFire, playerHurt, spottedIndex);
    std::cout << "netcode moments: " << netcodeResult.positionIndex.size() << std::endl;
    NonConsecutiveResult nonConsecutiveResult = queryNonConsecutive(position);
    std::cout << "nonconsecutive moments: " << nonConsecutiveResult.positionIndex.size() << std::endl;
    GroupingResult groupingResult = queryGrouping(position);
    std::cout << "grouping moments: " << groupingResult.positionIndex.size() << std::endl;

    CompoundAABB aroundARegions = {{
        {{462., 136., 0.}, {1860., 1257., 0.}},
        {{166., 2048., 0.}, {1955., 3168., 0.}},
        {{1141., 831., 0.}, {1940., 3200., 0.}}
    }};
    aroundARegions.regions[0].coverAllZ();
    aroundARegions.regions[1].coverAllZ();
    aroundARegions.regions[2].coverAllZ();
    CompoundAABB exactlyOnASite {{
        {{932., 2421., 0.}, {1260., 2653., 0.}},
        {{1028., 2318., 0.}, {1253., 2537., 0.}}
    }};
    exactlyOnASite.regions[0].coverAllZ();
    exactlyOnASite.regions[1].coverAllZ();
    GroupInSequenceOfRegionsResult successfulATakes =
            queryGroupingInSequenceOfRegions(position, groupingResult, {aroundARegions, exactlyOnASite},
                                             {true, true}, {true, false}, {TEAM_T});
    std::cout << "successful a takes moments: " << successfulATakes.positionIndex.size() << std::endl;
    GroupInSequenceOfRegionsResult failedATakes =
            queryGroupingInSequenceOfRegions(position, groupingResult, {aroundARegions, exactlyOnASite},
                                             {true, false}, {true, false}, {TEAM_T});
    std::cout << "failed a takes moments: " << failedATakes.positionIndex.size() << std::endl;
    std::cout << "total ticks: " << position.size << std::endl;
    vector<string> queryNames = {"velocity", "lookers", "wallers", "baiters", "netcode", "nonconsecutive", "grouping",
                                 "successful_A_takes", "failed_A_takes"};
    map<string, reference_wrapper<QueryResult>> queries {
        {queryNames[0], velocityResult},
        {queryNames[1], lookersResult},
        {queryNames[2], wallersResult},
        {queryNames[3], baitersResult},
        {queryNames[4], netcodeResult},
        {queryNames[5], nonConsecutiveResult},
        {queryNames[6], groupingResult},
        {queryNames[7], successfulATakes},
        {queryNames[8], failedATakes},
    };
     */
    /*
    vector<string> analysisNames = {aCatPeekersName, aCatSequenceName, midCTPeekersName, midCTSequenceName, lookerName};
    map<string, reference_wrapper<QueryResult>> analyses {
            {analysisNames[0], aCatPeekers},
            {analysisNames[1], aCatClusterSequence},
            {analysisNames[2], midCTPeekers},
            {analysisNames[3], midCTClusterSequence},
            {analysisNames[4], lookersResult},
    };
     */
    /*
    map<string, reference_wrapper<QueryResult>> analyses {
            //{lookerName, lookersResult},
            {engagementAimName, engagementAimResult},
            //{nearestOriginName,                 nearestOriginResult},
            //{playerInCoverEdgeName,             playerInCoverEdgeResult},
            //{teamLookingAtCoverEdgeClusterName, teamLookingAtCoverEdgeClusterResult},
    };

    // create the output files and the metadata describing files
    for (const auto & [name, result] : analyses) {
        std::ofstream fsTimed;
        std::cout << "writing " << outputDir + "/" + timestamp + "_" + name + ".csv" << std::endl;
        fsTimed.open(outputDir + "/" + timestamp + "_" + name + ".csv");
        fsTimed << result.get().toCSV();
        fsTimed.close();
        std::ofstream fsOverride;
        std::cout << "writing " << outputDir + "/" + name + ".csv" << std::endl;
        fsOverride.open(outputDir + "/" + name + ".csv");
        fsOverride << result.get().toCSV();
        fsOverride.close();
    }
    */

        /*
        // data sets describes types of data sets
        // versions describes all versions, with most recent first
        std::fstream datasetsTest, datasets, versions;
        string datasetsPath = outputDir + "/datasets.txt";
        string versionsPath = outputDir + "/versions.txt";
        datasetsTest.open(datasetsPath);
        bool createdFiles = datasetsTest.good();
        datasetsTest.close();
        if (createdFiles) {
            versions.open(versionsPath, std::fstream::in);
            std::stringstream oldVersionsContents;
            oldVersionsContents << versions.rdbuf();
            versions.close();
            versions.open(versionsPath, std::fstream::out | std::fstream::trunc);
            versions << timestamp << std::endl;
            versions << oldVersionsContents.rdbuf();
        }
        else {
            datasets.open(datasetsPath, std::fstream::out);
            for (const auto & [name, query] : analyses) {
                datasets << name << std::endl;
                for (int i = 0; i < query.get().keysForDiff.size(); i++) {
                    if (i != 0) {
                        datasets << ",";

                    }
                    datasets << query.get().keysForDiff[i];
                }
                datasets << std::endl;
            }
            datasets.close();
            versions.open(versionsPath, std::fstream::out);
            versions << timestamp;
            versions.close();
        }
         */

    map<string, reference_wrapper<QueryResult>> analyses {
        {"rounds", queryRounds},
        {"weaponFire", queryWeaponFire},
        {"hurt", queryHurt},
        {"kills", queryKills},
        {inferenceLatentOrderName, inferenceLatentOrderResult},
        {latentEngagementName, latentEngagementResult},
        {inferenceLatentEngagementName, inferenceLatentEngagementResult},
        {inferenceLatentAggressionName, inferenceLatentAggressionResult},
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
    d2DistanceToPlacesResult.writeAreasPerPlace(outputDir + "/" + "areas_per_place.csv");

    //vector<string> queryNames = {"games", "rounds", "players", "ticks", "playerAtTick", "aCatClusterSequence", "aCatClusters", "midCTClusterSequence", "midTClusters", "lookers"};
    map<string, reference_wrapper<QueryResult>> queries {
            {"games", queryGames},
            {"rounds", queryRounds},
            {"players", queryPlayers},
            {"ticks", queryTicks},
            {"playerAtTick", queryPlayerAtTick},
            {"smokeGrenades", smokeGrenadeResult},
            {"playerFlashed", playerFlashedResult},
            {dust2MeshName, d2MeshResult},
            {dust2CellsName, d2CellsResult},
            {dust2AreaVisibleName, d2AreaVisibleResult},
            {dust2CellVisibleName, d2CellVisibleResult},
            {dust2DangerName, d2NavDangerResult},
            {dust2ReachableName, d2ReachableResult},
            {dust2DistanceToPlacesName, d2DistanceToPlacesResult},
            //{aggressionEventName, aggressionEventResult},
            {engagementName, engagementResult},
            /* aim model queries
            {engagementPerTickAimName, engagementPerTickAimResult},
            {inferenceEngagementAimName, inferenceEngagementAimResult},
             */
            {nonEngagementTrajectoryName, nonEngagementTrajectoryResult},
            {trajectorySegmentName, trajectorySegmentResult},
            {inferenceLatentOrderName, inferenceLatentOrderResult},
            {inferenceLatentOrderDistributionName, inferenceLatentOrderDistributionResult},
            {latentEngagementName, latentEngagementResult},
            {inferenceLatentEngagementName, inferenceLatentEngagementResult},
            {inferenceLatentEngagementDistributionName, inferenceLatentEngagementDistributionResult},
            {inferenceLatentAggressionName, inferenceLatentAggressionResult},
            {inferenceLatentAggressionDistributionName, inferenceLatentAggressionDistributionResult},
            //{inferenceLatentEngagementHitName, inferenceLatentEngagementHitResult},
            //{inferenceLatentEngagementVisibleName, inferenceLatentEngagementVisibleResult},
            //{queryNames[5], aCatClusterSequence},
            //{queryNames[6], aCatPeekersClusters},
            //{queryNames[7], midCTClusterSequence},
            //{queryNames[8], midCTPeekersClusters},
            //{queryNames[9], lookersResult}
    };

    if (runServer) {
        std::cout << "starting server" << std::endl;
        httplib::Server svr;
        svr.set_default_headers({
            {"Access-Control-Allow-Origin", "*"}
        });
        // Mount / to ./www directory
        auto ret = svr.set_mount_point("/nav/", navPath);
        svr.set_file_request_handler([](const httplib::Request &, httplib::Response &res) {
            res.set_header("Content-Encoding", "gzip");
        });
        svr.set_file_extension_and_mimetype_mapping("gz", "application/octet-stream");
        if (!ret) {
            // The specified base directory doesn't exist...
            throw std::runtime_error("nav directory doesn't exist");
        }

        svr.Get("/query/(\\w+)", [&](const httplib::Request & req, httplib::Response &res) {
            string resultType = req.matches[1];
            std::stringstream ss;
            if (queries.find(resultType) != queries.end()) {
                res.set_content(queries.find(resultType)->second.get().toCSV(), "text/csv");
            }
            else {
                res.status = 404;
            }
        });

        svr.Get("/query/(\\w+)/(\\d+)", [&](const httplib::Request & req, httplib::Response &res) {
            string resultType = req.matches[1];
            int64_t filter = std::stol(req.matches[2].str());
            if (queries.find(resultType) != queries.end()) {
                res.set_content(queries.find(resultType)->second.get().toCSV(filter), "text/csv");
            }
            else {
                res.status = 404;
            }
        });

        /*
        svr.Get("/games", [&](const httplib::Request & req, httplib::Response &res) {
            std::stringstream ss;
            for (int64_t gameIndex = 0; gameIndex < games.size; gameIndex++) {
                ss << games.id[gameIndex] << "," << games.demoFile[gameIndex] << ","
                    << games.demoTickRate[gameIndex];
                ss << std::endl;
            }
            res.set_content(ss.str(), "text/plain");
            res.set_header("Access-Control-Allow-Origin", "*");
        });

        svr.Get("/players/(\\d+)", [&](const httplib::Request & req, httplib::Response &res) {
            int64_t gameId = std::stol(req.matches[1].str());
            std::stringstream ss;
            for (int64_t playerIndex = games.playersPerGame->minId; playerIndex <= games.playersPerGame->maxId; playerIndex++) {
                ss << players.id[playerIndex] << "," << players.name[playerIndex] << std::endl;
            }
            res.set_content(ss.str(), "text/plain");
            res.set_header("Access-Control-Allow-Origin", "*");
        });

        svr.Get("/rounds/(\\d+)", [&](const httplib::Request & req, httplib::Response &res) {
            int64_t gameId = std::stol(req.matches[1].str());
            std::stringstream ss;
            for (int64_t roundIndex = games.roundsPerGame->minId; roundIndex <= games.roundsPerGame->maxId; roundIndex++) {
                ss << rounds.id[roundIndex] << "," << rounds.gameId[roundIndex] << "," << rounds.startTick[roundIndex]
                    << "," << rounds.endTick[roundIndex] << "," << rounds.warmup[roundIndex] << ","
                    << rounds.freezeTimeEnd[roundIndex] << "," << rounds.roundNumber[roundIndex] << ","
                    << rounds.roundEndReason[roundIndex] << "," << rounds.winner[roundIndex] << ","
                    << rounds.tWins[roundIndex] << "," << rounds.ctWins[roundIndex] << std::endl;
            }
            res.set_content(ss.str(), "text/plain");
            res.set_header("Access-Control-Allow-Origin", "*");
        });

        svr.Get("/ticks/(\\d+)", [&](const httplib::Request & req, httplib::Response &res) {
            int64_t roundId = std::stol(req.matches[1].str());
            std::stringstream ss;
            for (int64_t tickIndex = rounds.ticksPerRound->minId; tickIndex <= rounds.ticksPerRound->maxId; tickIndex++) {
                ss << ticks.id[tickIndex] << "," << ticks.roundId[tickIndex] << "," << ticks.gameTime[tickIndex] << ","
                    << ticks.demoTickNumber[tickIndex] << "," << ticks.gameTickNumber[tickIndex] << "," << ticks.bombCarrier[tickIndex] << ","
                    << ticks.bombX[tickIndex] << "," << ticks.bombY[tickIndex] << "," << ticks.bombZ[tickIndex] << std::endl;
            }
            res.set_content(ss.str(), "text/plain");
            res.set_header("Access-Control-Allow-Origin", "*");
        });

        svr.Get("/playerAtTick/(\\d+)", [&](const httplib::Request & req, httplib::Response &res) {
            int64_t tickId = std::stol(req.matches[1].str());
            std::stringstream ss;
            for (int64_t patIndex = ticks.patPerTick->minId; patIndex <= ticks.patPerTick->maxId; patIndex++) {
                ss << playerAtTick.id[patIndex] << "," << playerAtTick.playerId[patIndex] << "," << playerAtTick.tickId[patIndex] << ","
                   << playerAtTick.posX[patIndex] << "," << playerAtTick.posY[patIndex] << "," << playerAtTick.posZ[patIndex] << ","
                   << playerAtTick.viewX[patIndex] << "," << playerAtTick.viewY[patIndex] << "," << playerAtTick.team[patIndex] << ","
                   << playerAtTick.health[patIndex] << "," << playerAtTick.armor[patIndex] << "," << playerAtTick.isAlive[patIndex] << std::endl;
            }
            res.set_content(ss.str(), "text/plain");
            res.set_header("Access-Control-Allow-Origin", "*");
        });
         */

        // list schema is: name, num foreign keys, list of foreign key column names, other columns, other column names
        svr.Get("/list", [&](const httplib::Request &, httplib::Response &res) {
            std::stringstream ss;
            for (const auto & [queryName, _] : queries) {
                QueryResult & queryValue = queries.find(queryName)->second.get();
                ss << queryName << "," << queryValue.startTickColumn << "," << queryValue.getForeignKeyNames().size() << ",";
                for (const auto & keyName : queryValue.getForeignKeyNames()) {
                    ss << keyName << ",";
                }
                ss << queryValue.getOtherColumnNames().size() << ",";
                for (const auto & extraColName : queryValue.getOtherColumnNames()) {
                    ss << extraColName << ",";
                }
                if (queryValue.variableLength) {
                    ss << "c" << queryValue.perEventLengthColumn;
                }
                else {
                    ss << queryValue.ticksPerEvent;
                }
                ss << ",";
                bool firstKPC = true;
                for (const auto keyPlayerColumn : queryValue.keyPlayerColumns) {
                    if (!firstKPC) {
                        ss << ";";
                    }
                    ss << keyPlayerColumn;
                    firstKPC = false;
                }
                ss << ",";
                ss << boolToString(queryValue.nonTemporal);
                ss << ",";
                ss << boolToString(queryValue.overlay);
                ss << ",";
                ss << queryValue.overlayLabelsQuery;
                ss << ",";
                ss << boolToString(queryValue.havePlayerLabels);
                ss << ",";
                ss << queryValue.playersToLabelColumn;
                ss << ",";
                ss << queryValue.playerLabelIndicesColumn;
                ss << ",";
                QueryResult::commaSeparateList(ss, queryValue.playerLabels, ";");
                ss << ",";
                ss << queryValue.perTickPlayerLabels;
                ss << ",";
                ss << queryValue.perTickPlayerLabelsQuery;
                ss << ",";
                QueryResult::commaSeparateList(ss, queryValue.posLabelsPositions, ";");
                ss << ",";
                ss << queryValue.perTickPosLabels;
                ss << ",";
                ss << queryValue.perTickPosLabelsQuery;
                ss << ",";
                ss << boolToString(queryValue.havePerTickAimTable);
                ss << ",";
                ss << queryValue.perTickAimTable;
                ss << ",";
                ss << boolToString(queryValue.havePerTickAimPredictionTable);
                ss << ",";
                ss << queryValue.perTickPredictionAimTable;
                ss << ",";
                ss << queryValue.eventIdColumn;
                ss << ",";
                ss << queryValue.haveBlob;
                ss << ",";
                ss << queryValue.blobFileName;
                ss << ",";
                ss << queryValue.blobBytesPerRow;
                ss << ",";
                ss << queryValue.blobTotalBytes;
                ss << std::endl;
            }
            res.set_content(ss.str(), "text/plain");
        });

        svr.listen("0.0.0.0", 3123);
    }
    /*
    while (true) {
        usleep(1e6);
    }
     */
    return 0;
}