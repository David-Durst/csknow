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
#include "queries/velocity.h"
#include "queries/wallers.h"
#include "queries/baiters.h"
#include "queries/netcode.h"
#include "queries/looking.h"
#include "queries/nonconsecutive.h"
#include "queries/grouping.h"
#include "queries/groupInSequenceOfRegions.h"
#include "indices/spotted.h"
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"

using std::map;
using std::string;
using std::reference_wrapper;

int main(int argc, char * argv[]) {
    if (argc != 3 && argc != 4) {
        std::cout << "please call this code 2 arguments: " << std::endl;
        std::cout << "1. path/to/local_data" << std::endl;
        std::cout << "2. run server (y or n)" << std::endl;
        std::cout << "(optional) 3. path/to/output/dir" << std::endl;
        return 1;
    }
    string dataPath = argv[1];
    bool runServer = argv[2][0] == 'y';
    bool writeOutput = false;
    string outputDir = "";
    if (argc == 4) {
        outputDir = argv[3];
        writeOutput = true;
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
    Rounds rounds;
    Ticks ticks;
    PlayerAtTick playerAtTick;
    Spotted spotted;
    WeaponFire weaponFire;
    Kills kills;
    Hurt hurt;
    Grenades grenades;
    Flashed flashed;
    GrenadeTrajectories grenadeTrajectories;
    Plants plants;
    Defusals defusals;
    Explosions explosions;
    
    loadData(equipment, gameTypes, hitGroups, games, players, rounds, ticks, playerAtTick, spotted, weaponFire,
             kills, hurt, grenades, flashed, grenadeTrajectories, plants, defusals, explosions, dataPath);
    buildIndexes(equipment, gameTypes, hitGroups, games, players, rounds, ticks, playerAtTick, spotted, weaponFire,
                 kills, hurt, grenades, flashed, grenadeTrajectories, plants, defusals, explosions);
    //std::printf("GLIBCXX: %d\n",__GLIBCXX__);
    std::cout << "num elements in equipment: " << equipment.size << std::endl;
    std::cout << "num elements in game_types: " << gameTypes.size << std::endl;
    std::cout << "num elements in hit_groups: " << hitGroups.size << std::endl;
    std::cout << "num elements in games: " << games.size << std::endl;
    std::cout << "num elements in players: " << players.size << std::endl;
    std::cout << "num elements in rounds: " << rounds.size << std::endl;
    std::cout << "num elements in ticks: " << ticks.size << std::endl;
    std::cout << "num elements in playerAtTick: " << playerAtTick.size << std::endl;
    std::cout << "num elements in spotted: " << spotted.size << std::endl;
    std::cout << "num elements in weaponFire: " << weaponFire.size << std::endl;
    std::cout << "num elements in kills: " << kills.size << std::endl;
    std::cout << "num elements in hurt: " << hurt.size << std::endl;
    std::cout << "num elements in grenades: " << grenades.size << std::endl;
    std::cout << "num elements in flashed: " << flashed.size << std::endl;
    std::cout << "num elements in grenadeTrajectories: " << grenadeTrajectories.size << std::endl;
    std::cout << "num elements in plants: " << plants.size << std::endl;
    std::cout << "num elements in defusals: " << defusals.size << std::endl;
    std::cout << "num elements in explosions: " << explosions.size << std::endl;

    /*
    SpottedIndex spottedIndex(position, spotted);
    std::cout << "built spotted index" << std::endl;
     */

    /*
    VelocityResult velocityResult = queryVelocity(position);
    std::cout << "velocity moments: " << velocityResult.positionIndex.size() << std::endl;
    LookingResult lookersResult = queryLookers(position);
    std::cout << "looker moments: " << lookersResult.positionIndex.size() << std::endl;
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

    // create the output files and the metadata describing files
    /*
    if (writeOutput) {
        for (const auto & [name, result] : queries) {
            std::fstream fs;
            fs.open(outputDir + "/" + timestamp + "_" + name + ".csv", std::fstream::out);
            fs << result.get().toCSV(position);
            fs.close();
        }

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
            for (const auto & [name, query] : queries) {
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
    }
     */

    if (runServer) {
        httplib::Server svr;
        /*
        svr.Get("/query/(\\w+)", [&](const httplib::Request & req, httplib::Response &res) {
            string resultType = req.matches[1];
            res.set_header("Access-Control-Allow-Origin", "*");
            if (queries.find(resultType) != queries.end()) {
                res.set_content(queries.find(resultType)->second.get().toCSV(position), "text/csv");
            }
            else {
                res.status = 404;
            }
        });

        svr.Get("/query/(\\w+)/(.+).csv", [&](const httplib::Request & req, httplib::Response &res) {
            string resultType = req.matches[1];
            string game = req.matches[2];
            res.set_header("Access-Control-Allow-Origin", "*");
            if (queries.find(resultType) != queries.end()) {
                res.set_content(queries.find(resultType)->second.get().toCSVFiltered(position, game), "text/csv");
            }
            else {
                res.status = 404;
            }
        });
         */

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

        // list schema is: d/q (d for dataset, q for query), 
        svr.Get("/list", [&](const httplib::Request & req, httplib::Response &res) {
            std::stringstream ss;
            /*
            for (const auto queryName : queryNames) {
                QueryResult & queryValue = queries.find(queryName)->second.get();
                ss << queryName << "," << queryValue.getKeyNames().size() << ",";
                for (const auto & keyName : queryValue.getKeyNames()) {
                    ss << keyName << ",";
                }
                ss << queryValue.getExtraColumnNames().size() << ",";
                for (const auto & extraColName : queryValue.getExtraColumnNames()) {
                    ss << extraColName << ",";
                }
                ss << queryValue.getDatatype() << ",";
                if (queryValue.variableLength) {
                    ss << "c" << queryValue.ticksColumn;
                }
                else {
                    ss << queryValue.ticksPerEvent;
                }
                ss << std::endl;
            }
             */
            res.set_content(ss.str(), "text/plain");
            res.set_header("Access-Control-Allow-Origin", "*");
        });

        svr.listen("0.0.0.0", 3123);
    }
    /*
    while (true) {
        usleep(1e6);
    }
     */
}