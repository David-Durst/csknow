#include "load_data.h"
#include "file_helpers.h"
#include "fast_float/fast_float.h"
#include <algorithm>
#include <iostream>
#include <dirent.h>
#include <string>
#include <iostream>
#include <atomic>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <charconv>
#include <omp.h>

using std::to_string;
using std::string;
using std::string_view;


void loadEquipmentFile(Equipment & equipment, string filePath) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = 0;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, equipment.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, &equipment.name[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 2;
    }
    closeMMapFile({fd, stats, file});
}

void loadEquipment(Equipment & equipment, string dataPath) {
    string fileName = "dimension_table_equipment.csv";
    string filePath = dataPath + "/" + fileName;

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows({filePath});
    int64_t rows = startingPointPerFile[1];

    std::cout << "allocating arrays" << std::endl;
    equipment.init(rows, 1, startingPointPerFile);

    std::cout << "loading equipment off disk" << std::endl;
    equipment.fileNames[0] = fileName;
    loadEquipmentFile(equipment, filePath);
}

void loadGameTypesFile(GameTypes & gameTypes, string filePath) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = 0;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, gameTypes.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, &gameTypes.tableType[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 2;
    }
    closeMMapFile({fd, stats, file});
}

void loadGameTypes(GameTypes & gameTypes, string dataPath) {
    string fileName = "dimension_table_game_types.csv";
    string filePath = dataPath + "/" + fileName;

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows({filePath});
    int64_t rows = startingPointPerFile[1];

    std::cout << "allocating arrays" << std::endl;
    gameTypes.init(rows, 1, startingPointPerFile);

    std::cout << "loading game_types off disk" << std::endl;
    gameTypes.fileNames[0] = fileName;
    loadGameTypesFile(gameTypes, filePath);
}

void loadHitGroupsFile(HitGroups & hitGroups, string filePath) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = 0;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, hitGroups.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, &hitGroups.groupName[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 2;
    }
    closeMMapFile({fd, stats, file});
}

void loadHitGroups(HitGroups & hitGroups, string dataPath) {
    string fileName = "dimension_table_hit_groups.csv";
    string filePath = dataPath + "/" + fileName;

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows({filePath});
    int64_t rows = startingPointPerFile[1];

    std::cout << "allocating arrays" << std::endl;
    hitGroups.init(rows, 1, startingPointPerFile);

    std::cout << "loading hit_groups off disk" << std::endl;
    hitGroups.fileNames[0] = fileName;
    loadHitGroupsFile(hitGroups, filePath);
}

void loadGamesFile(Games & games, string filePath) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = 0;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, games.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, &games.demoFile[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, games.demoTickRate[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, games.gameTickRate[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, &games.mapName[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, games.gameType[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 6;
    }
    closeMMapFile({fd, stats, file});
}

void loadGames(Games & games, string dataPath) {
    string fileName = "global_games.csv";
    string filePath = dataPath + "/" + fileName;

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows({filePath});
    int64_t rows = startingPointPerFile[1];

    std::cout << "allocating arrays" << std::endl;
    games.init(rows, 1, startingPointPerFile);

    std::cout << "loading games off disk" << std::endl;
    games.fileNames[0] = fileName;
    loadGamesFile(games, filePath);
}

void loadPlayersFile(Players & players, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, players.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, players.gameId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, &players.name[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, players.steamId[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 4;
    }
    closeMMapFile({fd, stats, file});
}

void loadPlayers(Players & players, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/players", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    players.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading players off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadPlayersFile(players, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadRoundsFile(Rounds & rounds, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, rounds.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, rounds.gameId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, rounds.startTick[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, rounds.endTick[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, rounds.warmup[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, rounds.freezeTimeEnd[arrayEntry]);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, rounds.roundNumber[arrayEntry]);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, rounds.roundEndReason[arrayEntry]);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, rounds.winner[arrayEntry]);
        }
        else if (colNumber == 9) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, rounds.tWins[arrayEntry]);
        }
        else if (colNumber == 10) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, rounds.ctWins[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 11;
    }
    closeMMapFile({fd, stats, file});
}

void loadRounds(Rounds & rounds, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/rounds", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    rounds.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading rounds off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadRoundsFile(rounds, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadTicksFile(Ticks & ticks, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, ticks.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, ticks.roundId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, ticks.gameTime[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, ticks.demoTickNumber[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, ticks.gameTickNumber[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, ticks.bombCarrier[arrayEntry]);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, ticks.bombX[arrayEntry]);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, ticks.bombY[arrayEntry]);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, ticks.bombZ[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 9;
    }
    closeMMapFile({fd, stats, file});
}

void loadTicks(Ticks & ticks, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/ticks", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    ticks.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading ticks off disk" << std::endl;
    std::atomic<int> filesProcessed = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadTicksFile(ticks, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / filePaths.size());
    }
}

void loadPlayerAtTickFile(PlayerAtTick & pat, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.playerId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.tickId[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.posX[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.posY[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.posZ[arrayEntry]);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.viewX[arrayEntry]);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.viewY[arrayEntry]);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.team[arrayEntry]);
        }
        else if (colNumber == 9) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.health[arrayEntry]);
        }
        else if (colNumber == 10) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.armor[arrayEntry]);
        }
        else if (colNumber == 11) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.hasHelmet[arrayEntry]);
        }
        else if (colNumber == 12) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.isAlive[arrayEntry]);
        }
        else if (colNumber == 13) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.isCrouching[arrayEntry]);
        }
        else if (colNumber == 14) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.isAirborne[arrayEntry]);
        }
        else if (colNumber == 15) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.remainingFlashTime[arrayEntry]);
        }
        else if (colNumber == 16) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.activeWeapon[arrayEntry]);
        }
        else if (colNumber == 17) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.primaryWeapon[arrayEntry]);
        }
        else if (colNumber == 18) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.primaryBulletsClip[arrayEntry]);
        }
        else if (colNumber == 19) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.primaryBulletsReserve[arrayEntry]);
        }
        else if (colNumber == 20) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.secondaryWeapon[arrayEntry]);
        }
        else if (colNumber == 21) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.secondaryBulletsClip[arrayEntry]);
        }
        else if (colNumber == 22) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.secondaryBulletsReserve[arrayEntry]);
        }
        else if (colNumber == 23) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.numHe[arrayEntry]);
        }
        else if (colNumber == 24) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.numFlash[arrayEntry]);
        }
        else if (colNumber == 25) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.numSmoke[arrayEntry]);
        }
        else if (colNumber == 26) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.numMolotov[arrayEntry]);
        }
        else if (colNumber == 27) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.numIncendiary[arrayEntry]);
        }
        else if (colNumber == 28) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.numDecoy[arrayEntry]);
        }
        else if (colNumber == 29) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.numZeus[arrayEntry]);
        }
        else if (colNumber == 30) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.hasDefuser[arrayEntry]);
        }
        else if (colNumber == 31) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.hasBomb[arrayEntry]);
        }
        else if (colNumber == 32) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.money[arrayEntry]);
        }
        else if (colNumber == 33) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, pat.ping[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 34;
    }
    closeMMapFile({fd, stats, file});
}

void loadPlayerAtTick(PlayerAtTick & pat, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/player_at_tick", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    pat.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading player_at_tick off disk" << std::endl;
    std::atomic<int> filesProcessed = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadPlayerAtTickFile(pat, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / filePaths.size());
    }

    pat.makePitchNeg90To90();
}

void loadSpottedFile(Spotted & spotted, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.tickId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.spottedPlayer[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.spotterPlayer[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.isSpotted[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 5;
    }
    closeMMapFile({fd, stats, file});
}

void loadSpotted(Spotted & spotted, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/spotted", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    spotted.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading spotted off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadSpottedFile(spotted, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadWeaponFireFile(WeaponFire & weaponFire, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, weaponFire.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, weaponFire.tickId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, weaponFire.shooter[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, weaponFire.weapon[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 4;
    }
    closeMMapFile({fd, stats, file});
}

void loadWeaponFire(WeaponFire & weaponFire, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/weapon_fire", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    weaponFire.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading weapon_fire off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadWeaponFireFile(weaponFire, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadKillsFile(Kills & kills, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.tickId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.killer[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.victim[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.weapon[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.assister[arrayEntry]);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.isHeadshot[arrayEntry]);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.isWallbang[arrayEntry]);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.penetratedObjects[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 9;
    }
    closeMMapFile({fd, stats, file});
}

void loadKills(Kills & kills, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/kills", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    kills.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading kills off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadKillsFile(kills, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadHurtFile(Hurt & hurt, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, hurt.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, hurt.tickId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, hurt.victim[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, hurt.attacker[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, hurt.weapon[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, hurt.armorDamage[arrayEntry]);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, hurt.armor[arrayEntry]);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, hurt.healthDamage[arrayEntry]);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, hurt.health[arrayEntry]);
        }
        else if (colNumber == 9) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, hurt.hitGroup[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 10;
    }
    closeMMapFile({fd, stats, file});
}

void loadHurt(Hurt & hurt, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/hurt", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    hurt.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading hurt off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadHurtFile(hurt, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadGrenadesFile(Grenades & grenades, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenades.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenades.thrower[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenades.grenadeType[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenades.throwTick[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenades.activeTick[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenades.expiredTick[arrayEntry]);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenades.destroyTick[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 7;
    }
    closeMMapFile({fd, stats, file});
}

void loadGrenades(Grenades & grenades, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/grenades", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    grenades.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading grenades off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadGrenadesFile(grenades, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadFlashedFile(Flashed & flashed, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, flashed.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, flashed.tickId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, flashed.grenadeId[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, flashed.thrower[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, flashed.victim[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 5;
    }
    closeMMapFile({fd, stats, file});
}

void loadFlashed(Flashed & flashed, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/flashed", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    flashed.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading flashed off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadFlashedFile(flashed, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadGrenadeTrajectoriesFile(GrenadeTrajectories & grenadeTrajectories, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenadeTrajectories.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenadeTrajectories.grenadeId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenadeTrajectories.idPerGrenade[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenadeTrajectories.posX[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenadeTrajectories.posY[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenadeTrajectories.posZ[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 6;
    }
    closeMMapFile({fd, stats, file});
}

void loadGrenadeTrajectories(GrenadeTrajectories & grenadeTrajectories, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/grenade_trajectories", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    grenadeTrajectories.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading grenade_trajectories off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadGrenadeTrajectoriesFile(grenadeTrajectories, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadPlantsFile(Plants & plants, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, plants.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, plants.startTick[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, plants.endTick[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, plants.planter[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, plants.succesful[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 5;
    }
    closeMMapFile({fd, stats, file});
}

void loadPlants(Plants & plants, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/plants", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    plants.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading grenade_trajectories off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadPlantsFile(plants, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadDefusalsFile(Defusals & defusals, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, defusals.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, defusals.plantId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, defusals.startTick[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, defusals.endTick[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, defusals.defuser[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, defusals.succesful[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 6;
    }
    closeMMapFile({fd, stats, file});
}

void loadDefusals(Defusals & defusals, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/defusals", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    defusals.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading defusals off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadDefusalsFile(defusals, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadExplosionsFile(Explosions & explosions, string filePath, int64_t fileRowStart, int32_t fileNumber) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = fileRowStart;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, explosions.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, explosions.plantId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, explosions.tickId[arrayEntry]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 3;
    }
    closeMMapFile({fd, stats, file});
}

void loadExplosions(Explosions & explosions, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/explosions", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    explosions.init(rows, filePaths.size(), startingPointPerFile);

    std::cout << "loading explosions off disk" << std::endl;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadExplosionsFile(explosions, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
    }
}

void loadData(Equipment & equipment, GameTypes & gameTypes, HitGroups & hitGroups, Games & games, Players & players,
              Rounds & rounds, Ticks & ticks, PlayerAtTick & playerAtTick, Spotted & spotted, WeaponFire & weaponFire,
              Kills & kills, Hurt & hurt, Grenades & grenades, Flashed & flashed, GrenadeTrajectories & grenadeTrajectories,
              Plants & plants, Defusals & defusals, Explosions & explosions, string dataPath) {
    std::cout << "loading equipment" << std::endl;
    loadEquipment(equipment, dataPath);
    std::cout << "loading game_types" << std::endl;
    loadGameTypes(gameTypes, dataPath);
    std::cout << "loading hit_groups" << std::endl;
    loadHitGroups(hitGroups, dataPath);
    std::cout << "loading games" << std::endl;
    loadGames(games, dataPath);
    std::cout << "loading players" << std::endl;
    loadPlayers(players, dataPath);
    std::cout << "loading rounds" << std::endl;
    loadRounds(rounds, dataPath);
    std::cout << "loading ticks" << std::endl;
    loadTicks(ticks, dataPath);
    std::cout << "loading player_at_tick" << std::endl;
    loadPlayerAtTick(playerAtTick, dataPath);
    std::cout << "loading spotted" << std::endl;
    loadSpotted(spotted, dataPath);
    std::cout << "loading weaponFire" << std::endl;
    loadWeaponFire(weaponFire, dataPath);
    std::cout << "loading kills" << std::endl;
    loadKills(kills, dataPath);
    std::cout << "loading hurt" << std::endl;
    loadHurt(hurt, dataPath);
    std::cout << "loading grenades" << std::endl;
    loadGrenades(grenades, dataPath);
    std::cout << "loading flashed" << std::endl;
    loadFlashed(flashed, dataPath);
    std::cout << "loading grenadeTrajectories" << std::endl;
    loadGrenadeTrajectories(grenadeTrajectories, dataPath);
    std::cout << "loading plants" << std::endl;
    loadPlants(plants, dataPath);
    std::cout << "loading defusals" << std::endl;
    loadDefusals(defusals, dataPath);
    std::cout << "loading explosions" << std::endl;
    loadExplosions(explosions, dataPath);
}