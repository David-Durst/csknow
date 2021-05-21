#include "load_clusters.h"
#include "file_helpers.h"

Cluster::Cluster(string filePath) {
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
            this->id.push_back(0);
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, this->id[this->id.size() - 1]);
        }
        else if (colNumber == 1) {
            this->wallId.push_back(0);
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, this->wallId[this->wallId.size() - 1]);
        }
        else if (colNumber == 2) {
            this->x.push_back(0.0);
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, this->x[this->x.size() - 1]);
        }
        else if (colNumber == 3) {
            this->y.push_back(0);
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, this->y[this->y.size() - 1]);
        }
        else if (colNumber == 4) {
            this->z.push_back(0);
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, this->z[this->z.size() - 1]);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 5;
    }
    closeMMapFile({fd, stats, file});
}