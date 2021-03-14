#include "load_data.h"
#include "fast_float/fast_float.h"
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

using std::to_string;
using std::string;
using std::string_view;

void printProgress(double progress) {
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

const string placeholderFileName = ".placeholder";
void getFilesInDirectory(string path, vector<string> & files) {
    DIR * dir;
    struct dirent * en;
    dir = opendir(path.c_str());
    if (dir) {
        while ((en = readdir(dir)) != NULL) {
            if (en->d_type == DT_REG && placeholderFileName.compare(en->d_name) != 0) {
                files.push_back(path + "/" + en->d_name);
            }
        }
        closedir(dir);
    }
}

struct MMapFile {
    int fd;
    struct stat stats;
    const char * file;
};

MMapFile openMMapFile(string filePath) {
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd < 0)
    {
        fprintf(stderr, "Error opening file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    struct stat stats;
    fstat(fd, &stats);
    const char * file = (char *) mmap(NULL, stats.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    return {fd, stats, file};
}

void closeMMapFile(MMapFile mMapFile) {
    munmap((void *) mMapFile.file, mMapFile.stats.st_size);
    close(mMapFile.fd);
}

size_t getNextDelimiter(const char * file, size_t curEntryStart, size_t fileLength) {
    for (size_t curPos = curEntryStart; curPos < fileLength; curPos++) {
        if (file[curPos] == '\n' || file[curPos] == ',') {
            return curPos;
        }
    }
    return fileLength;
}

size_t getNewline(const char * file, size_t curEntryStart, size_t fileLength) {
    for (size_t curPos = curEntryStart; curPos < fileLength; curPos++) {
        if (file[curPos] == '\n') {
            return curPos;
        }
    }
    return fileLength;
}

int64_t getRows(string filePath) {
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd < 0)
    {
        fprintf(stderr, "Error opening file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    struct stat stats;
    fstat(fd, &stats);
    const char * file = (char *) mmap(NULL, stats.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    // -2 to skip header and because going to increment once at end of file
    int64_t numRows = -2;
    for (size_t size = 0; size < stats.st_size; numRows++, size = getNewline(file, size+1, stats.st_size)) ;

    munmap((void *) file, stats.st_size);
    close(fd);
    return numRows;
}

vector<int64_t> getFileStartingRows(vector<string> filePaths) {
    vector<int64_t> startingPointPerFile;
    startingPointPerFile.resize(filePaths.size()+1);
    std::atomic<int64_t> filesProcessed = 0;
    startingPointPerFile[0] = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        startingPointPerFile[fileIndex+1] = getRows(filePaths[fileIndex]);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / filePaths.size());
    }
    std::cout << std::endl;
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        startingPointPerFile[fileIndex+1] += startingPointPerFile[fileIndex];
    }
    return startingPointPerFile;
}

static inline __attribute__((always_inline))
void printParsingError(std::errc ec, int64_t rowNumber, int64_t colNumber) {
    if (ec != std::errc()) {
        std::cerr << "error parsing row " << rowNumber << " col " << colNumber << std::endl;
    }
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, double & value) {
    auto messages = fast_float::from_chars(&file[start], &file[end], value);
    printParsingError(messages.ec, rowNumber, colNumber);
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, int32_t & value) {
    auto messages = std::from_chars(&file[start], &file[end], value);
    printParsingError(messages.ec, rowNumber, colNumber);
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, int8_t & value) {
    auto messages = std::from_chars(&file[start], &file[end], value);
    printParsingError(messages.ec, rowNumber, colNumber);
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, char ** value) {
    *value = (char *) malloc ((end-start+1) * sizeof(char));
    strncpy(*value, &file[start], end-start);
    (*value)[end-start] = '\0';
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, bool & value) {
    int tmpVal;
    auto messages = std::from_chars(&file[start], &file[end], tmpVal);
    value = tmpVal != 0;
    printParsingError(messages.ec, rowNumber, colNumber);
}

void loadPositionFile(Position & position, string filePath, int64_t fileRowStart, int32_t fileNumber) {
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
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.demoTickNumber[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.gameTickNumber[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.matchStarted[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.gamePhase[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.roundsPlayed[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.isWarmup[arrayEntry]);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.roundStart[arrayEntry]);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.roundEnd[arrayEntry]);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.roundEndReason[arrayEntry]);
        }
        else if (colNumber == 9) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.freezeTimeEnded[arrayEntry]);
        }
        else if (colNumber == 10) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.tScore[arrayEntry]);
        }
        else if (colNumber == 11) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.ctScore[arrayEntry]);
        }
        else if (colNumber == 12) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, position.numPlayers[arrayEntry]);
        }
        // check last element before loop so don't need loop ending condition
        else if (colNumber == 103) {
            if (rowNumber == 0) {
                position.fileNames[fileNumber] = string(&file[curStart], curDelimiter-curStart);
            }
            position.demoFile[arrayEntry] = fileNumber;
            rowNumber++;
            arrayEntry++;
        }
        else if ((colNumber - 13) % 9 == 0) {
            readCol(file, curStart, curDelimiter, &position.players[(colNumber - 13) / 9].name[arrayEntry]);
        }
        else if ((colNumber - 13) % 9 == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber,
                    position.players[(colNumber - 13) / 9].team[arrayEntry]);
        }
        else if ((colNumber - 13) % 9 == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber,
                    position.players[(colNumber - 13) / 9].xPosition[arrayEntry]);
        }
        else if ((colNumber - 13) % 9 == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber,
                    position.players[(colNumber - 13) / 9].yPosition[arrayEntry]);
        }
        else if ((colNumber - 13) % 9 == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber,
                    position.players[(colNumber - 13) / 9].zPosition[arrayEntry]);
        }
        else if ((colNumber - 13) % 9 == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber,
                    position.players[(colNumber - 13) / 9].xViewDirection[arrayEntry]);
        }
        else if ((colNumber - 13) % 9 == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber,
                    position.players[(colNumber - 13) / 9].yViewDirection[arrayEntry]);
        }
        else if ((colNumber - 13) % 9 == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber,
                    position.players[(colNumber - 13) / 9].isAlive[arrayEntry]);
        }
        else if ((colNumber - 13) % 9 == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber,
                    position.players[(colNumber - 13) / 9].isBlinded[arrayEntry]);
        }
        colNumber = (colNumber + 1) % 104;
    }
    closeMMapFile({fd, stats, file});
}

void loadPositions(Position & position, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/position", filePaths);

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    //std::cout << "rows: " << rows << std::endl;
    //std::cout << "sizeof row: " << sizeof(int32_t)*3 + sizeof(bool)*5 + sizeof(int8_t)*6 + 10*(sizeof(double)*5 + 5*sizeof(char) + sizeof(char*) + sizeof(int8_t)) << std::endl;
    position.init(rows, filePaths.size());

    std::cout << "loading positions off disk" << std::endl;
    std::atomic<int> filesProcessed = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadPositionFile(position, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / filePaths.size());
    }
    std::cout << std::endl;
    std::cout << "finished" << std::endl;
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
            readCol(file, curStart, curDelimiter, &spotted.spottedPlayer[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, &spotted.player0Name[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.player0Spotter[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, &spotted.player1Name[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.player1Spotter[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, &spotted.player2Name[arrayEntry]);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.player2Spotter[arrayEntry]);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, &spotted.player3Name[arrayEntry]);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.player3Spotter[arrayEntry]);
        }
        else if (colNumber == 9) {
            readCol(file, curStart, curDelimiter, &spotted.player4Name[arrayEntry]);
        }
        else if (colNumber == 10) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.player4Spotter[arrayEntry]);
        }
        else if (colNumber == 11) {
            readCol(file, curStart, curDelimiter, &spotted.player5Name[arrayEntry]);
        }
        else if (colNumber == 12) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.player5Spotter[arrayEntry]);
        }
        else if (colNumber == 13) {
            readCol(file, curStart, curDelimiter, &spotted.player6Name[arrayEntry]);
        }
        else if (colNumber == 14) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.player6Spotter[arrayEntry]);
        }
        else if (colNumber == 15) {
            readCol(file, curStart, curDelimiter, &spotted.player7Name[arrayEntry]);
        }
        else if (colNumber == 16) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.player7Spotter[arrayEntry]);
        }
        else if (colNumber == 17) {
            readCol(file, curStart, curDelimiter, &spotted.player8Name[arrayEntry]);
        }
        else if (colNumber == 18) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.player8Spotter[arrayEntry]);
        }
        else if (colNumber == 19) {
            readCol(file, curStart, curDelimiter, &spotted.player9Name[arrayEntry]);
        }
        else if (colNumber == 20) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.player9Spotter[arrayEntry]);
        }
        else if (colNumber == 21) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, spotted.demoTickNumber[arrayEntry]);
        }
        else if (colNumber == 22) {
            if (rowNumber == 0) {
                spotted.fileNames[fileNumber] = string(&file[curStart], curDelimiter-curStart);
            }
            spotted.demoFile[arrayEntry] = fileNumber;
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 23;
    }
    closeMMapFile({fd, stats, file});
}

void loadSpotted(Spotted & spotted, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/spotted", filePaths);

    std::cout << "determining array size" << std::endl;
    vector startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    spotted.init(rows, filePaths.size());

    std::cout << "loading positions off disk" << std::endl;
    std::atomic<int> filesProcessed = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadSpottedFile(spotted, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / filePaths.size());
    }
    std::cout << std::endl;
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
            readCol(file, curStart, curDelimiter, &weaponFire.shooter[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, &weaponFire.weapon[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, weaponFire.demoTickNumber[arrayEntry]);
        }
        else if (colNumber == 3) {
            if (rowNumber == 0) {
                weaponFire.fileNames[fileNumber] = string(&file[curStart], curDelimiter-curStart);
            }
            weaponFire.demoFile[arrayEntry] = fileNumber;
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
    vector startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    weaponFire.init(rows, filePaths.size());

    std::cout << "loading positions off disk" << std::endl;
    std::atomic<int> filesProcessed = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadWeaponFireFile(weaponFire, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / filePaths.size());
    }
    std::cout << std::endl;
}

void loadPlayerHurtFile(PlayerHurt & playerHurt, string filePath, int64_t fileRowStart, int32_t fileNumber) {
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
            readCol(file, curStart, curDelimiter, &playerHurt.victimName[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, playerHurt.armorDamage[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, playerHurt.armor[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, playerHurt.healthDamage[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, playerHurt.health[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, &playerHurt.attacker[arrayEntry]);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, &playerHurt.weapon[arrayEntry]);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, playerHurt.demoTickNumber[arrayEntry]);
        }
        else if (colNumber == 8) {
            if (rowNumber == 0) {
                playerHurt.fileNames[fileNumber] = string(&file[curStart], curDelimiter-curStart);
            }
            playerHurt.demoFile[arrayEntry] = fileNumber;
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 9;
    }
    closeMMapFile({fd, stats, file});
}

void loadPlayerHurt(PlayerHurt & playerHurt, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/hurt", filePaths);

    std::cout << "determining array size" << std::endl;
    vector startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    playerHurt.init(rows, filePaths.size());

    std::cout << "loading positions off disk" << std::endl;
    std::atomic<int> filesProcessed = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadPlayerHurtFile(playerHurt, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / filePaths.size());
    }
    std::cout << std::endl;
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
            readCol(file, curStart, curDelimiter, &grenades.thrower[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, &grenades.grenadeType[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, grenades.demoTickNumber[arrayEntry]);
        }
        else if (colNumber == 3) {
            if (rowNumber == 0) {
                grenades.fileNames[fileNumber] = string(&file[curStart], curDelimiter-curStart);
            }
            grenades.demoFile[arrayEntry] = fileNumber;
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 4;
    }
    closeMMapFile({fd, stats, file});
}

void loadGrenades(Grenades & grenades, string dataPath) {
    vector<string> filePaths;
    getFilesInDirectory(dataPath + "/grenades", filePaths);

    std::cout << "determining array size" << std::endl;
    vector startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    grenades.init(rows, filePaths.size());

    std::cout << "loading positions off disk" << std::endl;
    std::atomic<int> filesProcessed = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadGrenadesFile(grenades, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / filePaths.size());
    }
    std::cout << std::endl;
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
            readCol(file, curStart, curDelimiter, &kills.killer[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, &kills.victim[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, &kills.weapon[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, &kills.assister[arrayEntry]);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.isHeadshot[arrayEntry]);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.isWallbang[arrayEntry]);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.penetratedObjects[arrayEntry]);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, kills.demoTickNumber[arrayEntry]);
        }
        else if (colNumber == 8) {
            if (rowNumber == 0) {
                kills.fileNames[fileNumber] = string(&file[curStart], curDelimiter-curStart);
            }
            kills.demoFile[arrayEntry] = fileNumber;
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
    vector startingPointPerFile = getFileStartingRows(filePaths);
    int64_t rows = startingPointPerFile[filePaths.size()];

    std::cout << "allocating arrays" << std::endl;
    kills.init(rows, filePaths.size());

    std::cout << "loading positions off disk" << std::endl;
    std::atomic<int> filesProcessed = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        loadKillsFile(kills, filePaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / filePaths.size());
    }
    std::cout << std::endl;
}


void loadData(Position & position, Spotted & spotted, WeaponFire & weaponFire, PlayerHurt & playerHurt,
               Grenades & grenades, Kills & kills, string dataPath) {
    std::cout << "loading position" << std::endl;
    loadPositions(position, dataPath);
    std::cout << "loading spotted" << std::endl;
    loadSpotted(spotted, dataPath);
    std::cout << "loading weapon fire" << std::endl;
    loadWeaponFire(weaponFire, dataPath);
    std::cout << "loading player hurt" << std::endl;
    loadPlayerHurt(playerHurt, dataPath);
    std::cout << "loading grenades" << std::endl;
    loadGrenades(grenades, dataPath);
    std::cout << "loading kills" << std::endl;
    loadKills(kills, dataPath);
}