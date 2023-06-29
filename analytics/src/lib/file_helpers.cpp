#include "fast_float/fast_float.h"
#include "file_helpers.h"
#include <algorithm>
#include <iostream>
#include <dirent.h>
#include <string>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cerrno>
#include <vector>

using std::to_string;
using std::string;
using std::string_view;
using std::vector;

void printProgress(const atomic<int64_t> & numFinished, size_t numToComplete) {
    printProgress(static_cast<double>(numFinished.load()) / static_cast<double>(numToComplete));
}

void printProgress(const atomic<size_t> & numFinished, size_t numToComplete) {
    printProgress(static_cast<double>(numFinished.load()) / static_cast<double>(numToComplete));
}

void printProgress(size_t numFinished, size_t numToComplete) {
    printProgress(static_cast<double>(numFinished) / static_cast<double>(numToComplete));
}

void printProgress(double progress) {
    int barWidth = 70;

    std::cout << "[";
    int pos = static_cast<int>(barWidth * progress);
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(3) << progress * 100.0 << " %\r";
    std::cout.flush();
}

void getFilesInDirectory(const string & path, vector<string> & files) {
    DIR * dir;
    struct dirent * en;
    dir = opendir(path.c_str());
    if (dir) {
        while ((en = readdir(dir)) != nullptr) {
            if (en->d_type == DT_REG && placeholderFileName != en->d_name) {
                files.push_back(path + "/" + en->d_name);
            }
        }
        closedir(dir);
    }
    std::sort(files.begin(), files.end());
}

void createAndEmptyDirectory(const string & dir) {
    if (!fs::exists(dir)) {
        fs::create_directory(dir);
    }
    for (auto &path: fs::directory_iterator(dir)) {
        fs::remove(path);
    }
}

void removeDirectory(const string & dir) {
    for (auto &path: fs::directory_iterator(dir)) {
        fs::remove(path);
    }
    fs::remove(dir);
}

MMapFile openMMapFile(const string & filePath) {
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd < 0)
    {
        fprintf(stderr, "Error opening file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    struct stat stats;
    fstat(fd, &stats);
    const char * file = (char *) mmap(nullptr, stats.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
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

int64_t getRows(const string & filePath) {
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd < 0)
    {
        fprintf(stderr, "Error opening file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    struct stat stats;
    fstat(fd, &stats);
    const char * file = (char *) mmap(nullptr, stats.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    // -2 to skip header and because going to increment once at end of file
    int64_t numRows = -2;
    for (size_t size = 0; size < static_cast<size_t>(stats.st_size);
        numRows++, size = getNewline(file, size+1, stats.st_size)) ;

    munmap((void *) file, stats.st_size);
    close(fd);
    return numRows;
}

vector<int64_t> getFileStartingRows(const vector<string> & filePaths, bool printProgressBar) {
    vector<int64_t> startingPointPerFile;
    startingPointPerFile.resize(filePaths.size()+1);
    std::atomic<int64_t> filesProcessed = 0;
    startingPointPerFile[0] = 0;

    if (filePaths.size() > 1) {
#pragma omp parallel for
        for (size_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
            startingPointPerFile[fileIndex+1] = getRows(filePaths[fileIndex]);
            filesProcessed++;
            if (printProgressBar) {
                printProgress(static_cast<double>(filesProcessed.load()) / static_cast<double>(filePaths.size()));
            }
        }
    }
    else {
        for (size_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
            startingPointPerFile[fileIndex+1] = getRows(filePaths[fileIndex]);
            filesProcessed++;
            if (printProgressBar) {
                printProgress(static_cast<double>(filesProcessed.load()) / static_cast<double>(filePaths.size()));
            }
        }
    }
    if (printProgressBar) {
        std::cout << std::endl;
    }
    for (size_t fileIndex = 0; fileIndex < filePaths.size(); fileIndex++) {
        startingPointPerFile[fileIndex+1] += startingPointPerFile[fileIndex];
    }
    return startingPointPerFile;
}

std::vector<std::string> parseString(const std::string & input, char delimiter) {
    std::stringstream inputStream(input);
    std::vector<std::string> result;
    string token;
    while (getline(inputStream, token, delimiter)) {
        result.push_back(token);
    }
    return result;
}
