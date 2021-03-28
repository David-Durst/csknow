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

    Position position;
    Spotted spotted;
    WeaponFire weaponFire;
    PlayerHurt playerHurt;
    Grenades grenades;
    Kills kills;

    loadData(position, spotted, weaponFire, playerHurt, grenades, kills, dataPath);
    //std::printf("GLIBCXX: %d\n",__GLIBCXX__);
    std::cout << "num elements in position: " << position.size << std::endl;
    std::cout << "num elements in spotted: " << spotted.size << std::endl;
    std::cout << "num elements in weaponFire: " << weaponFire.size << std::endl;
    std::cout << "num elements in playerHurt: " << playerHurt.size << std::endl;
    std::cout << "num elements in grenades: " << grenades.size << std::endl;
    std::cout << "num elements in kills: " << kills.size << std::endl;

    SpottedIndex spottedIndex(position, spotted);
    std::cout << "built spotted index" << std::endl;

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
    std::cout << "total ticks: " << position.size << std::endl;
    vector<string> queryNames = {"velocity", "lookers", "wallers", "baiters", "netcode"};
    map<string, reference_wrapper<QueryResult>> queries {
        {queryNames[0], lookersResult},
        {queryNames[1], lookersResult},
        {queryNames[2], wallersResult},
        {queryNames[3], baitersResult},
        {queryNames[4], netcodeResult}
    };

    // create the output files and the metadata describing files
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

    if (runServer) {
        httplib::Server svr;
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

        svr.Get("/list", [&](const httplib::Request & req, httplib::Response &res) {
            std::stringstream ss;
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
                ss << queryValue.ticksPerEvent;
                ss << std::endl;
            }
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