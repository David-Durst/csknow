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
#include "load_cover.h"
#include "navmesh/nav_file.h"
#include "queries/nav_mesh.h"
#include "queries/train_dataset.h"
#include <filesystem>
namespace fs = std::filesystem;

using std::map;
using std::string;
using std::reference_wrapper;


int main(int argc, char * argv[]) {
    if (argc != 4) {
        std::cout << "please call this code 4 arguments: " << std::endl;
        std::cout << "1. path/to/local_data" << std::endl;
        std::cout << "2. path/to/nav_meshes" << std::endl;
        std::cout << "3. path/to/output/dir" << std::endl;
        return 1;
    }

    string dataPath = argv[1];
    string navPath = argv[2];
    string outputDir = argv[3];

    std::map<std::string, const nav_mesh::nav_file> map_navs;

    //Figure out from where to where you'd like to find a path
    for (const auto & entry : fs::directory_iterator(navPath)) {
        map_navs.insert(std::pair<std::string, nav_mesh::nav_file>(entry.path().filename().replace_extension(),
                                                                   nav_mesh::nav_file(entry.path().c_str())));
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
    CoverEdges coverEdges;
    CoverOrigins coverOrigins;

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

    TrainDatasetResult trainDatasetResult = queryTrainDataset(games, rounds, ticks, players, playerAtTick, map_navs);

    std::ofstream outputFile;
    string outputPath = outputDir + "/train_dataset.csv";

    std::cout << "writing train dataset with size " << trainDatasetResult.size << " to " << outputPath << std::endl;
    std::cout << trainDatasetResult.getDataLabelRanges() << std::endl;
    outputFile.open(outputPath);
    outputFile << trainDatasetResult.toCSV();
    outputFile.close();
}