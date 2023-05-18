//
// Created by durst on 5/17/23.
//
#include <iostream>
#include <unistd.h>
#include <map>
#include <string>
#include <sstream>
#include <functional>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include "load_data.h"

int main(int argc, char * argv[]) {
    if (argc != 3) {
        std::cout << "please call this code 2 arguments: " << std::endl;
        std::cout << "1. path/to/csv_folder" << std::endl;
        std::cout << "2. path/to/hdf5_file" << std::endl;
        return 1;
    }

    string csvPath = argv[1];
    string hdf5Path = argv[2];

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream timestampSS;
    timestampSS << std::put_time(&tm, "%d_%m_%Y__%H_%M_%S");
    string timestamp = timestampSS.str();
    std::cout << "timestamp: " << timestamp << std::endl;

    Equipment equipmentCSV, equipmentHDF5;
    GameTypes gameTypesCSV, gameTypesHDF5;
    HitGroups hitGroupsCSV, hitGroupsHDF5;
    Games gamesCSV, gamesHDF5;
    Players playersCSV, playersHDF5;
    Rounds unfilteredRoundsCSV(false), unfilteredRoundsHDF5(false),
        filteredRoundsCSV(true), filteredRoundsHDF5(true);
    Ticks ticksCSV, ticksHDF5;
    PlayerAtTick playerAtTickCSV, playerAtTickHDF5;
    Spotted spottedCSV, spottedHDF5;
    Footstep footstepCSV, footstepHDF5;
    WeaponFire weaponFireCSV, weaponFireHDF5;
    Kills killsCSV, killsHDF5;
    Hurt hurtCSV, hurtHDF5;
    Grenades grenadesCSV, grenadesHDF5;
    Flashed flashedCSV, flashedHDF5;
    GrenadeTrajectories grenadeTrajectoriesCSV, grenadeTrajectoriesHDF5;
    Plants plantsCSV, plantsHDF5;
    Defusals defusalsCSV, defusalsHDF5;
    Explosions explosionsCSV, explosionsHDF5;
    Say sayCSV, sayHDF5;

    auto csvReadStart = std::chrono::system_clock::now();
    loadDataCSV(equipmentCSV, gameTypesCSV, hitGroupsCSV, gamesCSV, playersCSV, unfilteredRoundsCSV, filteredRoundsCSV,
                ticksCSV, playerAtTickCSV, spottedCSV, footstepCSV, weaponFireCSV,
                killsCSV, hurtCSV, grenadesCSV, flashedCSV, grenadeTrajectoriesCSV, plantsCSV, defusalsCSV,
                explosionsCSV, sayCSV, csvPath);
    auto csvReadEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> csvReadTime = csvReadEnd - csvReadStart;
    std::cout << "csv read time " << csvReadTime.count() << std::endl;

    //std::printf("GLIBCXX: %d\n",__GLIBCXX__);
    std::cout << "num elements in equipment: " << equipmentCSV.size << std::endl;
    std::cout << "num elements in game_types: " << gameTypesCSV.size << std::endl;
    std::cout << "num elements in hit_groups: " << hitGroupsCSV.size << std::endl;
    std::cout << "num elements in games: " << gamesCSV.size << std::endl;
    std::cout << "num elements in players: " << playersCSV.size << std::endl;
    std::cout << "num elements in unfiltered_rounds: " << unfilteredRoundsCSV.size << std::endl;
    std::cout << "num elements in filtered_rounds: " << filteredRoundsCSV.size << std::endl;
    std::cout << "num elements in ticks: " << ticksCSV.size << std::endl;
    std::cout << "num elements in playerAtTick: " << playerAtTickCSV.size << std::endl;
    std::cout << "num elements in spotted: " << spottedCSV.size << std::endl;
    std::cout << "num elements in footstep: " << footstepCSV.size << std::endl;
    std::cout << "num elements in weaponFire: " << weaponFireCSV.size << std::endl;
    std::cout << "num elements in kills: " << killsCSV.size << std::endl;
    std::cout << "num elements in hurt: " << hurtCSV.size << std::endl;
    std::cout << "num elements in grenades: " << grenadesCSV.size << std::endl;
    std::cout << "num elements in flashed: " << flashedCSV.size << std::endl;
    std::cout << "num elements in grenadeTrajectories: " << grenadeTrajectoriesCSV.size << std::endl;
    std::cout << "num elements in plants: " << plantsCSV.size << std::endl;
    std::cout << "num elements in defusals: " << defusalsCSV.size << std::endl;
    std::cout << "num elements in explosions: " << explosionsCSV.size << std::endl;
    std::cout << "num elements in say: " << sayCSV.size << std::endl;

    auto hdf5WriteStart = std::chrono::system_clock::now();
    saveDataHDF5(equipmentCSV, gameTypesCSV, hitGroupsCSV, gamesCSV, playersCSV, unfilteredRoundsCSV, filteredRoundsCSV,
                ticksCSV, playerAtTickCSV, spottedCSV, footstepCSV, weaponFireCSV,
                killsCSV, hurtCSV, grenadesCSV, flashedCSV, grenadeTrajectoriesCSV, plantsCSV, defusalsCSV,
                explosionsCSV, sayCSV, hdf5Path);
    auto hdf5WriteEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> hdf5WriteTime = hdf5WriteEnd - hdf5WriteStart;
    std::cout << "hdf5 write time " << hdf5WriteTime.count() << std::endl;

    auto hdf5ReadStart = std::chrono::system_clock::now();
    loadDataHDF5(equipmentHDF5, gameTypesHDF5, hitGroupsHDF5, gamesHDF5, playersHDF5, unfilteredRoundsHDF5, filteredRoundsHDF5,
                 ticksHDF5, playerAtTickHDF5, spottedHDF5, footstepHDF5, weaponFireHDF5,
                 killsHDF5, hurtHDF5, grenadesHDF5, flashedHDF5, grenadeTrajectoriesHDF5, plantsHDF5, defusalsHDF5,
                 explosionsHDF5, sayHDF5, hdf5Path);
    auto hdf5ReadEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> hdf5ReadTime = hdf5ReadEnd - hdf5ReadStart;
    std::cout << "hdf5 read time " << hdf5ReadTime.count() << std::endl;

    std::cout << "testing equality" << std::endl;
    bool equipmentEqual = equipmentCSV == equipmentHDF5;
    std::cout << "equipment equal " << equipmentEqual << std::endl;
    bool gameTypesEqual = gameTypesCSV == gameTypesHDF5;
    std::cout << "game types equal " << gameTypesEqual << std::endl;
    bool hitGroupsEqual = hitGroupsCSV == hitGroupsHDF5;
    std::cout << "hit groups equal " << hitGroupsEqual << std::endl;
    bool playersEqual = playersCSV == playersHDF5;
    std::cout << "players equal " << playersEqual << std::endl;
    bool unfilteredRoundsEqual = unfilteredRoundsCSV == unfilteredRoundsHDF5;
    std::cout << "unfiltered rounds equal " << unfilteredRoundsEqual << std::endl;
    bool filteredRoundsEqual = filteredRoundsCSV == filteredRoundsHDF5;
    std::cout << "filtered rounds equal " << filteredRoundsEqual << std::endl;
    bool ticksEqual = ticksCSV == ticksHDF5;
    std::cout << "ticks equal " << ticksEqual << std::endl;
    bool playerAtTickEqual = playerAtTickCSV == playerAtTickHDF5;
    std::cout << "playerAtTick equal " << playerAtTickEqual << std::endl;
    bool spottedEqual = spottedCSV == spottedHDF5;
    std::cout << "spotted equal " << spottedEqual << std::endl;
    bool footstepEqual = footstepCSV == footstepHDF5;
    std::cout << "footstep equal " << footstepEqual << std::endl;
    bool weaponFireEqual = weaponFireCSV == weaponFireHDF5;
    std::cout << "weaponFire equal " << weaponFireEqual << std::endl;
    bool killsEqual = killsCSV == killsHDF5;
    std::cout << "kills equal " << killsEqual << std::endl;
    bool hurtEqual = hurtCSV == hurtHDF5;
    std::cout << "hurt equal " << hurtEqual << std::endl;
    bool grenadesEqual = grenadesCSV == grenadesHDF5;
    std::cout << "grenades equal " << grenadesEqual << std::endl;
    bool flashedEqual = flashedCSV == flashedHDF5;
    std::cout << "flashed equal " << flashedEqual << std::endl;
    bool grenadeTrajectoriesEqual = grenadeTrajectoriesCSV == grenadeTrajectoriesHDF5;
    std::cout << "grenadeTrajectories equal " << grenadeTrajectoriesEqual << std::endl;
    bool plantsEqual = plantsCSV == plantsHDF5;
    std::cout << "plants equal " << plantsEqual << std::endl;
    bool defusalsEqual = defusalsCSV == defusalsHDF5;
    std::cout << "defusals equal " << defusalsEqual << std::endl;
    bool explosionsEqual = explosionsCSV == explosionsHDF5;
    std::cout << "explosions equal " << explosionsEqual << std::endl;
    bool sayEqual = sayCSV == sayHDF5;
    std::cout << "say equal " << sayEqual << std::endl;
    std::cout << "all equal " << (equipmentEqual && gameTypesEqual && hitGroupsEqual && playersEqual &&
        unfilteredRoundsEqual && filteredRoundsEqual && ticksEqual && playerAtTickEqual &&
        spottedEqual && footstepEqual && weaponFireEqual && killsEqual && hurtEqual && grenadesEqual &&
        flashedEqual && grenadeTrajectoriesEqual && plantsEqual && defusalsEqual && explosionsEqual && sayEqual)
        << std::endl;
}