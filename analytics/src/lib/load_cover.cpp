//
// Created by durst on 12/29/21.
//
#include "load_cover.h"
#include "load_data.h"
#include "file_helpers.h"
#include <string>

void loadCoverEdgesFile(CoverEdges & coverEdges, string filePath) {
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
         curDelimiter < static_cast<size_t>(stats.st_size);
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverEdges.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverEdges.originId[arrayEntry]);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverEdges.clusterId[arrayEntry]);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverEdges.aabbs[arrayEntry].min.x);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverEdges.aabbs[arrayEntry].min.y);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverEdges.aabbs[arrayEntry].min.z);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverEdges.aabbs[arrayEntry].max.x);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverEdges.aabbs[arrayEntry].max.y);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverEdges.aabbs[arrayEntry].max.z);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 9;
    }
    closeMMapFile({fd, stats, file});
}

void loadCoverEdges(CoverEdges & coverEdges, string dataPath) {
    string fileName = "dimension_table_cover_edges.csv";
    string filePath = dataPath + "/" + fileName;

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows({filePath});
    int64_t rows = startingPointPerFile[1];

    std::cout << "allocating arrays" << std::endl;
    coverEdges.init(rows, 1);

    std::cout << "loading cover edges off disk" << std::endl;
    coverEdges.fileNames[0] = fileName;
    loadCoverEdgesFile(coverEdges, filePath);
}

void loadCoverOriginsFile(CoverOrigins & coverOrigins, string filePath) {
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
         curDelimiter < static_cast<size_t>(stats.st_size);
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverOrigins.id[arrayEntry]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverOrigins.origins[arrayEntry].x);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverOrigins.origins[arrayEntry].y);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, coverOrigins.origins[arrayEntry].z);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 4;
    }
    closeMMapFile({fd, stats, file});
}

void loadCoverOrigins(CoverOrigins & coverOrigins, string dataPath) {
    string fileName = "dimension_table_cover_origins.csv";
    string filePath = dataPath + "/" + fileName;

    std::cout << "determining array size" << std::endl;
    vector<int64_t> startingPointPerFile = getFileStartingRows({filePath});
    int64_t rows = startingPointPerFile[1];

    std::cout << "allocating arrays" << std::endl;
    coverOrigins.init(rows, 1);

    std::cout << "loading cover edges off disk" << std::endl;
    coverOrigins.fileNames[0] = fileName;
    loadCoverOriginsFile(coverOrigins, filePath);
}

void loadCover(CoverOrigins & origins, CoverEdges & edges, string dataPath) {
    std::cout << "loading cover edges" << std::endl;
    loadCoverEdges(edges, dataPath);
    std::cout << "loading cover origins" << std::endl;
    loadCoverOrigins(origins, dataPath);
}
