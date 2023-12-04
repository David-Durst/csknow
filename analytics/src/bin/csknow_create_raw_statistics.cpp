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
#include "queries/moments/feature_store_team_extractor.h"
#include "bots/analysis/humanness_metrics.h"
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
    if (argc != 3) {
        std::cout << "please call this code with 2 arguments: " << std::endl;
        std::cout << "1. path/to/local_data" << std::endl;
        std::cout << "2. path/to/output/dir" << std::endl;
        return 1;
    }

    string dataPath = argv[1];
    string outputDir = argv[2];

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

    int64_t numGames = games.size;
    int64_t numRounds = filteredRounds.size;
    int64_t numTicks = 0, numShots = 0, numKills = 0;
    for (int64_t roundIndex = 0; roundIndex < filteredRounds.size; roundIndex++) {
        int64_t gameId = filteredRounds.gameId[roundIndex];
        if (games.gameTickRate[gameId] < 125 || games.gameTickRate[gameId] > 130) {
            std::cout << "bad game id " << gameId << std::endl;
        }

        int64_t startTickIndex = filteredRounds.ticksPerRound[roundIndex].minId;
        int64_t endTickIndex = filteredRounds.ticksPerRound[roundIndex].maxId;
        numTicks += ticks.gameTickNumber[endTickIndex] - ticks.gameTickNumber[startTickIndex] + 1;

        for (int64_t tickIndex = startTickIndex; tickIndex <= endTickIndex; tickIndex++) {
            for (const auto &_: ticks.weaponFirePerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                (void) _;
                numShots++;
            }
            for (const auto &_: ticks.killsPerTick.intervalToEvent.findOverlapping(tickIndex, tickIndex)) {
                (void) _;
                numKills++;
            }
        }
    }

    std::ofstream fsStatistics;
    fsStatistics.open(outputDir + "/all_statistics.csv", std::fstream::app);
    fsStatistics << dataPath << "," << numGames << "," << numRounds << "," << numTicks << "," << numShots << "," << numKills << std::endl;
    std::ofstream fsPlayers;
    fsPlayers.open(outputDir + "/all_players.csv", std::fstream::app);
    for (int64_t i = 1; i < players.size; i++) {
        fsPlayers << players.name[i] << std::endl;
    }

    return 0;
}
