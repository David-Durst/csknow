#ifdef false
#include "load_regions.h"
#include <limits>

Regions loadRegions(string filePath) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(filePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track data for results
    Regions result;
    char ** name = (char **) malloc(sizeof(char*));;
    double minX, minY, minZ, maxX, maxY, maxZ;
    bool flip;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, name);
        }
        else if (colNumber == 1) {
            if (curStart == curDelimiter) {
                minX = -1 * std::numeric_limits<double>::max();
            }
            else {
                readCol(file, curStart, curDelimiter, rowNumber, colNumber, minX);
            }
        }
        else if (colNumber == 2) {
            if (curStart == curDelimiter) {
                minY = -1 * std::numeric_limits<double>::max();
            }
            else {
                readCol(file, curStart, curDelimiter, rowNumber, colNumber, minY);
            }
        }
        else if (colNumber == 3) {
            if (curStart == curDelimiter) {
                minZ = -1 * std::numeric_limits<double>::max();
            }
            else {
                readCol(file, curStart, curDelimiter, rowNumber, colNumber, minZ);
            }
        }
        else if (colNumber == 4) {
            if (curStart == curDelimiter) {
                maxX = std::numeric_limits<double>::max();
            }
            else {
                readCol(file, curStart, curDelimiter, rowNumber, colNumber, maxX);
            }
        }
        else if (colNumber == 5) {
            if (curStart == curDelimiter) {
                maxY = std::numeric_limits<double>::max();
            }
            else {
                readCol(file, curStart, curDelimiter, rowNumber, colNumber, maxY);
            }
        }
        else if (colNumber == 6) {
            if (curStart == curDelimiter) {
                maxZ = std::numeric_limits<double>::max();
            }
            else {
                readCol(file, curStart, curDelimiter, rowNumber, colNumber, maxZ);
            }
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, flip);
            result.id.push_back(rowNumber++);
            result.name.push_back(*name);
            result.aabb.push_back({{minX, minY, minZ}, {maxX, maxY, maxZ}});
            result.flip.push_back(flip);
        }
        colNumber = (colNumber + 1) % 8;
    }
    closeMMapFile({fd, stats, file});
    free(name);
    return result;
}

#endif //false