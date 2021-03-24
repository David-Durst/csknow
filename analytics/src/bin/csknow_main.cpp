#include <iostream>
#include <unistd.h>
#include <map>
#include <string>
#include <sstream>
#include <functional>
#include "load_data.h"
#include "queries/wallers.h"
#include "queries/baiters.h"
#include "queries/netcode.h"
#include "indices.h"
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"

using std::map;
using std::string;
using std::reference_wrapper;

int main(int argc, char * argv[]) {
    if (argc != 3) {
        std::cout << "please call this code 2 arguments: " << std::endl;
        std::cout << "1. path/to/local_data" << std::endl;
        std::cout << "2. run server (y or n)" << std::endl;
        return 1;
    }
    string dataPath = argv[1];
    bool runServer = argv[2][0] == 'y';

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

    WallersResult wallersResult = queryWallers(position, spotted);
    std::cout << "waller moments: " << wallersResult.positionIndex.size() << std::endl;
    BaitersResult baitersResult = queryBaiters(position, kills, spottedIndex);
    std::cout << "baiter moments: " << baitersResult.positionIndex.size() << std::endl;
    NetcodeResult netcodeResult = queryNetcode(position, weaponFire, playerHurt, spottedIndex);
    std::cout << "netcode moments: " << netcodeResult.positionIndex.size() << std::endl;
    std::cout << "total ticks: " << position.size << std::endl;
    vector<string> queryNames = {"wallers", "baiters", "netcode"};
    map<string, reference_wrapper<QueryResult>> queries {
        {queryNames[0], wallersResult},
        {queryNames[1], baitersResult},
        {queryNames[2], netcodeResult}
    };

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
                ss << queryName << "," << queryValue.getSourceName() << ","
                    << queryValue.getTargetNames().size() << ",";
                for (const auto & targetName : queryValue.getTargetNames()) {
                    ss << targetName << ",";
                }
                ss << queryValue.getExtraColumnNames().size() << ",";
                for (const auto & extraColName : queryValue.getExtraColumnNames()) {
                    ss << extraColName << ",";
                }
                ss << queryValue.getDatatype();
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