#include <iostream>
#include <unistd.h>
#include "load_data.h"
#include "queries/wallers.h"
#include "queries/baiters.h"
#include "indices.h"
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"

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
    std::cout << "total ticks: " << position.size << std::endl;

    if (runServer) {
        httplib::Server svr;
        svr.Get("/waller", [&](const httplib::Request &, httplib::Response &res) {
            res.set_content(wallersResult.toCSV(position), "text/csv");
            res.set_header("Access-Control-Allow-Origin", "*");
        });

        svr.Get("/waller/(.+).csv", [&](const httplib::Request & req, httplib::Response &res) {
            string game = req.matches[1];
            res.set_content(wallersResult.toCSVFiltered(position, game), "text/csv");
            res.set_header("Access-Control-Allow-Origin", "*");
        });

        svr.Get("/baiter", [&](const httplib::Request &, httplib::Response &res) {
            res.set_content(baitersResult.toCSV(position), "text/csv");
            res.set_header("Access-Control-Allow-Origin", "*");
        });

        svr.Get("/baiter/(.+).csv", [&](const httplib::Request & req, httplib::Response &res) {
            string game = req.matches[1];
            res.set_content(baitersResult.toCSVFiltered(position, game), "text/csv");
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