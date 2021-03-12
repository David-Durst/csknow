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

    loadData(position, spotted, weaponFire, playerHurt, grenades, kills, dataPath);
    std::cout << "num elements in position: " << position.size << std::endl;
}