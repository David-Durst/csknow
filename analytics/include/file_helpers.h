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
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

using std::to_string;
using std::string;
using std::string_view;
using std::vector;
using std::atomic;

void printProgress(const atomic<int64_t> & numFinished, size_t numToComplete);
void printProgress(const atomic<size_t> & numFinished, size_t numToComplete);
void printProgress(double progress);

const string placeholderFileName = ".placeholder";
void getFilesInDirectory(const string & path, vector<string> & files);

void createAndEmptyDirectory(const string & dir);

struct MMapFile {
    int fd;
    struct stat stats;
    const char * file;
};

MMapFile openMMapFile(const string & filePath);

void closeMMapFile(MMapFile mMapFile);

size_t getNextDelimiter(const char * file, size_t curEntryStart, size_t fileLength);

size_t getNewline(const char * file, size_t curEntryStart, size_t fileLength);

int64_t getRows(const string & filePath);

vector<int64_t> getFileStartingRows(const vector<string> & filePaths, bool printProgressBar = false) ;

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
void readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber, float & value) {
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
    // need to append null terminator as none is applied
    // by default in strncpy
    (*value)[end-start] = '\0';
}

static inline __attribute__((always_inline))
void readCol(const char * file, size_t start, size_t end, string & value) {
    value.assign(&file[start], end-start);
}

static inline __attribute__((always_inline))
bool readCol(const char * file, size_t start, size_t end, int64_t rowNumber, int64_t colNumber) {
    int tmpVal;
    auto messages = std::from_chars(&file[start], &file[end], tmpVal);
    printParsingError(messages.ec, rowNumber, colNumber);
    return tmpVal != 0;
}

std::vector<std::string> parseString(const std::string & input, char delimiter);

#endif //CSKNOW_FILE_HELPERS_H
