#ifndef CSKNOW_FILE_HELPERS_H
#define CSKNOW_FILE_HELPERS_H

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
    std::sort(files.begin(), files.end());
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

vector<int64_t> getFileStartingRows(vector<string> filePaths, bool printProgressBar = false) {
    vector<int64_t> startingPointPerFile;
    startingPointPerFile.resize(filePaths.size()+1);
    std::atomic<int64_t> filesProcessed = 0;
    startingPointPerFile[0] = 0;
#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        startingPointPerFile[fileIndex+1] = getRows(filePaths[fileIndex]);
        filesProcessed++;
        if (printProgressBar) {
            printProgress((filesProcessed * 1.0) / filePaths.size());
        }
    }
    if (printProgressBar) {
        std::cout << std::endl;
    }
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
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, int64_t & value) {
    if (file[start] == '\\' && file[start+1] == 'N') {
        value = -1;
        return;
    }
    auto messages = std::from_chars(&file[start], &file[end], value);
    printParsingError(messages.ec, rowNumber, colNumber);
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, int32_t & value) {
    if (file[start] == '\\' && file[start+1] == 'N') {
        value = -1;
        return;
    }
    auto messages = std::from_chars(&file[start], &file[end], value);
    printParsingError(messages.ec, rowNumber, colNumber);
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, int16_t & value) {
    if (file[start] == '\\' && file[start+1] == 'N') {
        value = -1;
        return;
    }
    auto messages = std::from_chars(&file[start], &file[end], value);
    printParsingError(messages.ec, rowNumber, colNumber);
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, int8_t & value) {
    if (file[start] == '\\' && file[start+1] == 'N') {
        value = -1;
        return;
    }
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

#endif //CSKNOW_FILE_HELPERS_H
