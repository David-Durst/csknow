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
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd < 0)
    {
        fprintf(stderr, "Error opening file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    struct stat stats;
    fstat(fd, &stats);
    const char * file = (char *) mmap(NULL, stats.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

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
    munmap((void *) file, stats.st_size);
    close(fd);
}

void loadPositions(Position & position, string dataPath) {
    vector<string> positionPaths = getFilesInDirectory(dataPath + "/position");

    std::cout << "determining array size" << std::endl;
    int64_t startingPointPerFile[positionPaths.size()+1];
    std::atomic<int64_t> filesProcessed = 0;
    startingPointPerFile[0] = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < positionPaths.size(); fileIndex++) {
        startingPointPerFile[fileIndex+1] = getRows(positionPaths[fileIndex]);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / positionPaths.size());
    }
    std::cout << std::endl;
    for (int64_t fileIndex = 0; fileIndex < positionPaths.size(); fileIndex++) {
        startingPointPerFile[fileIndex+1] += startingPointPerFile[fileIndex];
    }
    int64_t rows = startingPointPerFile[positionPaths.size()];

    std::cout << "allocating arrays" << std::endl;
    std::cout << "rows: " << rows << std::endl;
    std::cout << "sizeof row: " << sizeof(int32_t)*3 + sizeof(bool)*7 + sizeof(int8_t)*7 + sizeof(double)*5 + 5*sizeof(char) + sizeof(char*) << std::endl;
    position.size = rows;
    position.fileNames.resize(positionPaths.size());
    position.demoTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
    while (true) {
        usleep(1e6);
    }
    position.gameTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
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
        position.players[i].name = (char **) malloc(rows * sizeof(char*));
        position.players[i].team = (int8_t *) malloc(rows * sizeof(int8_t));
        position.players[i].xPosition = (double *) malloc(rows * sizeof(double));
        position.players[i].yPosition = (double *) malloc(rows * sizeof(double));
        position.players[i].zPosition = (double *) malloc(rows * sizeof(double));
        position.players[i].xViewDirection = (double *) malloc(rows * sizeof(double));
        position.players[i].yViewDirection = (double *) malloc(rows * sizeof(double));
        position.players[i].isAlive = (bool *) malloc(rows * sizeof(bool));
        position.players[i].isBlinded = (bool *) malloc(rows * sizeof(bool));
    }
    position.demoFile = (int32_t *) malloc(rows * sizeof(int32_t*));

    std::cout << "loading positions off disk" << std::endl;
    filesProcessed = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < positionPaths.size(); fileIndex++) {
        loadPositionFile(position, positionPaths[fileIndex], startingPointPerFile[fileIndex], fileIndex);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / positionPaths.size());
    }
    std::cout << std::endl;
}

void loadData(Position & position, Spotted & spotted, WeaponFire & weaponFire, PlayerHurt & playerHurt,
               Grenades & grenades, Kills & kills, string dataPath) {
    loadPositions(position, dataPath);
}