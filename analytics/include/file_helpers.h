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

using std::to_string;
using std::string;
using std::string_view;
using std::vector;

void printProgress(double progress);

const string placeholderFileName = ".placeholder";
void getFilesInDirectory(string path, vector<string> & files);

struct MMapFile {
    int fd;
    struct stat stats;
    const char * file;
};

MMapFile openMMapFile(string filePath);

void closeMMapFile(MMapFile mMapFile);

size_t getNextDelimiter(const char * file, size_t curEntryStart, size_t fileLength);

size_t getNewline(const char * file, size_t curEntryStart, size_t fileLength);

int64_t getRows(string filePath);

vector<int64_t> getFileStartingRows(vector<string> filePaths, bool printProgressBar = false) ;

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
