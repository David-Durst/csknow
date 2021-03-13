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
vector<string> getFilesInDirectory(string path) {
    vector<string> result;
    DIR * dir;
    struct dirent * en;
    dir = opendir(path.c_str());
    if (dir) {
        while ((en = readdir(dir)) != NULL) {
            if (en->d_type == DT_REG && placeholderFileName.compare(en->d_name) != 0) {
                result.push_back(path + "/" + en->d_name);
            }
        }
        closedir(dir);
    }
    return result;
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
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, int64_t & value) {
    auto messages = std::from_chars(&file[start], &file[end], value);
    printParsingError(messages.ec, rowNumber, colNumber);
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, int8_t & value) {
    auto messages = std::from_chars(&file[start], &file[end], value);
    printParsingError(messages.ec, rowNumber, colNumber);
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, string & value) {
    value = string(&file[start], end-start);
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, bool & value) {
    int tmpVal;
    auto messages = std::from_chars(&file[start], &file[end], tmpVal);
    value = tmpVal != 0;
    printParsingError(messages.ec, rowNumber, colNumber);
}

void loadPositionFile(PositionBuilder & positionBuilder, string filePath) {
    // mmap the file
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd < 0)
    {
        fprintf(stderr, "Error opening file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    struct stat stats;
    fstat(fd, &stats);
    const char * file = (char *) mmap(NULL, stats.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    string_view row;

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // values we will parse into and then push
    double valueDouble;
    int64_t valueInt64;
    int8_t valueInt8;
    string valueString;
    bool valueBool;
    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
        curDelimiter < stats.st_size;
        curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueInt64);
            positionBuilder.demoTickNumber.push_back(valueInt64);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueInt64);
            positionBuilder.gameTickNumber.push_back(valueInt64);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueBool);
            positionBuilder.matchStarted.push_back(valueBool);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueInt8);
            positionBuilder.gamePhase.push_back(valueInt8);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueInt8);
            positionBuilder.roundsPlayed.push_back(valueInt8);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueBool);
            positionBuilder.isWarmup.push_back(valueBool);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueBool);
            positionBuilder.roundStart.push_back(valueBool);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueBool);
            positionBuilder.roundEnd.push_back(valueBool);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueInt8);
            positionBuilder.roundEndReason.push_back(valueInt8);
        }
        else if (colNumber == 9) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueBool);
            positionBuilder.freezeTimeEnded.push_back(valueBool);
        }
        else if (colNumber == 10) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueInt8);
            positionBuilder.tScore.push_back(valueInt8);
        }
        else if (colNumber == 11) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueInt8);
            positionBuilder.ctScore.push_back(valueInt8);
        }
        else if (colNumber == 12) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueInt8);
            positionBuilder.numPlayers.push_back(valueInt8);
        }
        // check last element before loop so don't need loop ending condition
        else if (colNumber == 103) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueString);
            positionBuilder.demoFile.push_back(valueString);
            rowNumber++;
        }
        else if ((colNumber - 13) % 9 == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueString);
            positionBuilder.players[(colNumber - 13) / 9].name.push_back(valueString);
        }
        else if ((colNumber - 13) % 9 == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueInt8);
            positionBuilder.players[(colNumber - 13) / 9].team.push_back(valueInt8);
        }
        else if ((colNumber - 13) % 9 == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueDouble);
            positionBuilder.players[(colNumber - 13) / 9].xPosition.push_back(valueDouble);
        }
        else if ((colNumber - 13) % 9 == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueDouble);
            positionBuilder.players[(colNumber - 13) / 9].yPosition.push_back(valueDouble);
        }
        else if ((colNumber - 13) % 9 == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueDouble);
            positionBuilder.players[(colNumber - 13) / 9].zPosition.push_back(valueDouble);
        }
        else if ((colNumber - 13) % 9 == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueDouble);
            positionBuilder.players[(colNumber - 13) / 9].xViewDirection.push_back(valueDouble);
        }
        else if ((colNumber - 13) % 9 == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueDouble);
            positionBuilder.players[(colNumber - 13) / 9].yViewDirection.push_back(valueDouble);
        }
        else if ((colNumber - 13) % 9 == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueBool);
            positionBuilder.players[(colNumber - 13) / 9].isAlive.push_back(valueBool);
        }
        else if ((colNumber - 13) % 9 == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, valueBool);
            positionBuilder.players[(colNumber - 13) / 9].isBlinded.push_back(valueBool);
        }
        colNumber = (colNumber + 1) % 104;
    }
    munmap((void *) file, stats.st_size);
    close(fd);
}

void loadPositions(Position & position, OpenFiles & openFiles, string dataPath) {

    std::cout << "loading positions off disk" << std::endl;

    vector<string> positionPaths = getFilesInDirectory(dataPath + "/position");

    vector<PositionBuilder> positions{positionPaths.size()};
    std::atomic<int64_t> filesProcessed = 0;
    openFiles.paths.clear();
    #pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < positionPaths.size(); fileIndex++) {
        openFiles.paths.insert(positionPaths[fileIndex]);
        loadPositionFile(positions[fileIndex], positionPaths[fileIndex]);
        openFiles.paths.erase(positionPaths[fileIndex]);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / positionPaths.size());
    }
    std::cout << std::endl;

    vector<int64_t> startingPointPerFile;
    std::cout << "allocating vectors" << std::endl;
    startingPointPerFile.push_back(0);
    for (int64_t fileIndex = 0; fileIndex < positionPaths.size(); fileIndex++) {
        startingPointPerFile.push_back(startingPointPerFile[fileIndex] + positions[fileIndex].demoTickNumber.size());
    }
    int64_t rows = startingPointPerFile[startingPointPerFile.size() - 1];
    position.size = rows;
    position.demoTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
    position.gameTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
    position.demoFile = (string *) malloc(rows * sizeof(string));
    position.matchStarted = (bool *) malloc(rows * sizeof(bool));
    position.gamePhase = (int8_t *) malloc(rows * sizeof(int8_t));
    position.roundsPlayed = (int8_t *) malloc(rows * sizeof(int8_t));
    position.isWarmup = (bool *) malloc(rows * sizeof(bool));
    position.roundStart = (bool *) malloc(rows * sizeof(bool));
    position.roundEnd = (bool *) malloc(rows * sizeof(bool));
    position.roundEndReason = (int8_t *) malloc(rows * sizeof(int8_t));
    position.freezeTimeEnded = (bool *) malloc(rows * sizeof(bool));
    position.tScore = (int8_t *) malloc(rows * sizeof(int8_t));
    position.ctScore = (int8_t *) malloc(rows * sizeof(int8_t));
    position.numPlayers = (int8_t *) malloc(rows * sizeof(int8_t));
    for (int i = 0; i < NUM_PLAYERS; i++) {
        position.players[i].name = (string *) malloc(rows * sizeof(string));
        position.players[i].team = (int8_t *) malloc(rows * sizeof(int8_t));
        position.players[i].xPosition = (double *) malloc(rows * sizeof(double));
        position.players[i].yPosition = (double *) malloc(rows * sizeof(double));
        position.players[i].zPosition = (double *) malloc(rows * sizeof(double));
        position.players[i].xViewDirection = (double *) malloc(rows * sizeof(double));
        position.players[i].yViewDirection = (double *) malloc(rows * sizeof(double));
        position.players[i].isAlive = (bool *) malloc(rows * sizeof(bool));
        position.players[i].isBlinded = (bool *) malloc(rows * sizeof(bool));
    }

    std::cout << "merging vectors" << std::endl;
    filesProcessed = 0;
    #pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < positionPaths.size(); fileIndex++) {
        int64_t fileRow = 0;
        for (int64_t positionIndex = startingPointPerFile[fileIndex]; positionIndex < startingPointPerFile[fileIndex-1];
             positionIndex++) {
            fileRow++;
            position.demoTickNumber[positionIndex] = positions[fileIndex].demoTickNumber[fileRow];
            position.gameTickNumber[positionIndex] = positions[fileIndex].gameTickNumber[fileRow];
            position.demoFile[positionIndex] = positions[fileIndex].demoFile[fileRow];
            position.matchStarted[positionIndex] = positions[fileIndex].matchStarted[fileRow];
            position.gamePhase[positionIndex] = positions[fileIndex].gamePhase[fileRow];
            position.roundsPlayed[positionIndex] = positions[fileIndex].roundsPlayed[fileRow];
            position.isWarmup[positionIndex] = positions[fileIndex].isWarmup[fileRow];
            position.roundStart[positionIndex] = positions[fileIndex].roundStart[fileRow];
            position.roundEnd[positionIndex] = positions[fileIndex].roundEnd[fileRow];
            position.roundEndReason[positionIndex] = positions[fileIndex].roundEndReason[fileRow];
            position.freezeTimeEnded[positionIndex] = positions[fileIndex].freezeTimeEnded[fileRow];
            position.tScore[positionIndex] = positions[fileIndex].tScore[fileRow];
            position.ctScore[positionIndex] = positions[fileIndex].ctScore[fileRow];
            position.numPlayers[positionIndex] = positions[fileIndex].numPlayers[fileRow];
            for (int i = 0; i < NUM_PLAYERS; i++) {
                position.players[i].name[positionIndex] = positions[fileIndex].players[i].name[fileRow];
                position.players[i].team[positionIndex] = positions[fileIndex].players[i].team[fileRow];
                position.players[i].xPosition[positionIndex] = positions[fileIndex].players[i].xPosition[fileRow];
                position.players[i].yPosition[positionIndex] = positions[fileIndex].players[i].yPosition[fileRow];
                position.players[i].zPosition[positionIndex] = positions[fileIndex].players[i].zPosition[fileRow];
                position.players[i].xViewDirection[positionIndex] = positions[fileIndex].players[i].xViewDirection[fileRow];
                position.players[i].yViewDirection[positionIndex] = positions[fileIndex].players[i].yViewDirection[fileRow];
                position.players[i].isAlive[positionIndex] = positions[fileIndex].players[i].isAlive[fileRow];
                position.players[i].isBlinded[positionIndex] = positions[fileIndex].players[i].isBlinded[fileRow];
            }
        }
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / positionPaths.size());
    }
    std::cout << std::endl;
}

void loadData(Position & position, Spotted & spotted, WeaponFire & weaponFire, PlayerHurt & playerHurt,
               Grenades & grenades, Kills & kills, string dataPath, OpenFiles & openFiles) {
    loadPositions(position, openFiles, dataPath);
    /*
     */
}