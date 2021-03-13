#include <iostream>
#include "load_data.h"

int main(int argc, char * argv[]) {
    if (argc != 2) {
        std::cout << "please call this code 1 argument: " << std::endl;
        std::cout << "1. path/to/local_data" << std::endl;
        return 1;
    }
    string dataPath = argv[1];

    Position position;
    Spotted spotted;
    WeaponFire weaponFire;
    PlayerHurt playerHurt;
    Grenades grenades;
    Kills kills;
    OpenFiles openFiles;

    try {
        loadData(position, spotted, weaponFire, playerHurt, grenades, kills, dataPath, openFiles);
    }
    catch (std::error_code& e) {
        for (const auto & path : openFiles.paths) {
            std::cerr << "open file: " << path << std::endl;
        }
        std::cerr << e.message() << std::endl;
        return 0;
    }
    //std::printf("GLIBCXX: %d\n",__GLIBCXX__);
    std::cout << "num elements in position: " << position.size << std::endl;
}