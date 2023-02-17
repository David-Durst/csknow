#ifndef CSKNOW_QUERY_H
#define CSKNOW_QUERY_H
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <functional>
#include "load_data.h"
#include "linear_algebra.h"
#include <highfive/H5File.hpp>
using std::vector;
using std::stringstream;
using std::string;

static inline __attribute__((always_inline))
string boolToString(bool val) {
    if (val) {
        return "true";
    }
    else {
        return "false";
    }
}

static inline __attribute__((always_inline))
uint32_t boolToInt(bool val) {
    if (val) {
        return 1;
    }
    else {
        return 0;
    }
}

/*
enum DataType {
    noSrcTarget = 0,
    justSrc = 1,
    justTarget = 2,
    srcAndTarget = 3
};
 */

static inline
string toSignedIntString(int64_t i, bool dropZero = false) {
    if (dropZero && i == 0) {
        return "";
    }
    if (i < 0) {
        return std::to_string(i);
    }
    else {
        return "+" + std::to_string(i);

    }
}

class QueryResult {
public:
//    vector<int64_t> positionIndex;
    // these are offsets relative to foreign key columns
    int startTickColumn;
    int ticksPerEvent;
    int perEventLengthColumn;
    bool variableLength = false;
    bool nonTemporal = false;
    bool overlay = false;
    string overlayLabelsQuery = "";
    bool havePlayerLabels = false;
    // these are offsets relative to other columns start
    int playersToLabelColumn = 0;
    int playerLabelIndicesColumn = 0;
    vector<string> playerLabels;
    bool havePerTickAimTable = false;
    string perTickAimTable;
    bool havePerTickAimPredictionTable = false;
    string perTickPredictionAimTable;
    int eventIdColumn = -1;
    bool haveBlob = false;
    string blobFileName = "";
    int blobBytesPerRow = INVALID_ID;
    int64_t blobTotalBytes = INVALID_ID;
    string extension = ".query";
    vector<int> keyPlayerColumns = {};
    string hdf5Prefix = "/data/";
    static HighFive::DataSetCreateProps defaultHDF5CreateProps;
//    vector<int> keysForDiff;

    //virtual string toCSVFiltered(const Position & position, string game) = 0;
    int64_t size = 0;

    void save(const string & mapsPath, const string & mapName) {
        string fileName = mapName + extension;
        string filePath = mapsPath + "/" + fileName;

        std::ofstream fsQuery(filePath, std::ios::trunc);
        fsQuery << toCSV();
        fsQuery.close();
    }

    string toCSV() {
        std::stringstream ss;
        addHeader(ss);
        for (int64_t index = 0; index < size; index++) {
            oneLineToCSV(index, ss);
        }
        return ss.str();
    }

    void toCSV(std::ostream & s) {
        addHeader(s);
        for (int64_t index = 0; index < size; index++) {
            oneLineToCSV(index, s);
        }
    }

    string toCSV(int64_t otherTableIndex) {
        std::stringstream ss;
        addHeader(ss);
        vector<int64_t> filter = filterByForeignKey(otherTableIndex);
        for (const auto & index : filter) {
            oneLineToCSV(index, ss);
        }
        return ss.str();
    }

    // TODO: Remove implementation, make all queries implement this
    virtual void toHDF5Inner(HighFive::File &, const HighFive::DataSetCreateProps &) {
        throw std::runtime_error("HDFS saving not implemented for this query yet");
    }

    void toHDF5(const string& filePath) {
        // We create an empty HDF55 file, by truncating an existing
        // file if required:
        HighFive::File file(filePath, HighFive::File::Overwrite);

        // create id column
        vector<int64_t> id_to_save;
        id_to_save.reserve(size);
        for (int64_t i = 0; i < size; i++) {
            id_to_save.push_back(i);
        }
        HighFive::DataSetCreateProps hdf5CreateProps;
        hdf5CreateProps.add(HighFive::Deflate(9));
        hdf5CreateProps.add(HighFive::Chunking(id_to_save.size()));
        file.createDataSet(hdf5Prefix + "id", id_to_save, hdf5CreateProps);

        // create all other columns
        toHDF5Inner(file, hdf5CreateProps);
    }

    void addHeader(std::ostream & s) {
        s << "id";
        for (const auto & foreignKey : getForeignKeyNames()) {
            s << "," << foreignKey;
        }
        for (const auto & otherCol : getOtherColumnNames()) {
            s << "," << otherCol;
        }
        s << std::endl;
    }

    std::vector<string> getHeader() {
        std::vector<string> result{"id"};
        for (const auto & foreignKey : getForeignKeyNames()) {
            result.push_back(foreignKey);
        }
        for (const auto & otherCol : getOtherColumnNames()) {
            result.push_back(otherCol);
        }
        return result;
    }

    static
    void commaSeparateList(std::ostream & s, vector<string> list, const string& separator = ",") {
        if (list.empty()) {
            return;
        }
        s << list[0];
        for (size_t i = 1; i < list.size(); i++) {
            s << separator << list[i];
        }
    }

    // find all rows with foreign key that reference another table
    virtual vector<int64_t> filterByForeignKey(int64_t otherTableIndex) = 0;
    virtual void oneLineToCSV(int64_t index, std::ostream &s) = 0;
    virtual vector<string> getForeignKeyNames() = 0;
    virtual vector<string> getOtherColumnNames() = 0;
};


template <typename T>
std::vector<int> vectorOfEnumsToVectorOfInts(const std::vector<T> & vectorOfEnums) {
    std::vector<int> result;
    result.reserve(vectorOfEnums.size());

    for (size_t i = 0; i < vectorOfEnums.size(); i++) {
        result.push_back(enumAsInt(vectorOfEnums[i]));
    }

    return result;
}

template <typename T, std::size_t N>
std::array<std::vector<T>, N> vectorOfArraysToArrayOfVectors(const std::vector<std::array<T, N>> & vectorOfArrays) {
    std::array<std::vector<T>, N> arrayOfVectors;
    for (size_t i = 0; i < arrayOfVectors.size(); i++) {
        arrayOfVectors[i].reserve(vectorOfArrays.size());
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfArrays.size(); vectorIndex++) {
        for (size_t arrayIndex = 0; arrayIndex < arrayOfVectors.size(); arrayIndex++) {
            arrayOfVectors[arrayIndex].push_back(vectorOfArrays[vectorIndex][arrayIndex]);
        }
    }

    return arrayOfVectors;
}

template <typename T, std::size_t N>
std::array<vector<int>, N> vectorOfEnumArraysToArrayOfIntVectors(const std::vector<std::array<T, N>> & vectorOfArrays) {
    std::array<std::vector<int>, N> arrayOfVectors;
    for (size_t i = 0; i < arrayOfVectors.size(); i++) {
        arrayOfVectors[i].reserve(vectorOfArrays.size());
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfArrays.size(); vectorIndex++) {
        for (size_t arrayIndex = 0; arrayIndex < arrayOfVectors.size(); arrayIndex++) {
            arrayOfVectors[arrayIndex].push_back(enumAsInt(vectorOfArrays[vectorIndex][arrayIndex]));
        }
    }

    return arrayOfVectors;
}

template <std::size_t N>
std::array<std::array<std::vector<double>, 2>, N>
vectorOfVec2ArraysToArrayOfArrayOfVectors(const std::vector<std::array<Vec2, N>> & vectorOfVec2Arrays){
    std::array<std::array<std::vector<double>, 2>, N> arrayOfArrayOfVectors;
    for (size_t i = 0; i < arrayOfArrayOfVectors.size(); i++) {
        for (size_t j = 0; j < 2; j++) {
            arrayOfArrayOfVectors[i][j].reserve(vectorOfVec2Arrays.size());
        }
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfVec2Arrays.size(); vectorIndex++) {
        for (size_t arrayIndex = 0; arrayIndex < arrayOfArrayOfVectors.size(); arrayIndex++) {
            arrayOfArrayOfVectors[arrayIndex][0].push_back(vectorOfVec2Arrays[vectorIndex][arrayIndex].x);
            arrayOfArrayOfVectors[arrayIndex][1].push_back(vectorOfVec2Arrays[vectorIndex][arrayIndex].y);
        }
    }

    return arrayOfArrayOfVectors;
}

template <std::size_t N>
std::array<std::array<std::vector<double>, 3>, N>
vectorOfVec3ArraysToArrayOfArrayOfVectors(const std::vector<std::array<Vec3, N>> & vectorOfVec3Arrays){
    std::array<std::array<std::vector<double>, 3>, N> arrayOfArrayOfVectors;
    for (size_t i = 0; i < arrayOfArrayOfVectors.size(); i++) {
        for (size_t j = 0; j < 3; j++) {
            arrayOfArrayOfVectors[i][j].reserve(vectorOfVec3Arrays.size());
        }
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfVec3Arrays.size(); vectorIndex++) {
        for (size_t arrayIndex = 0; arrayIndex < arrayOfArrayOfVectors.size(); arrayIndex++) {
            arrayOfArrayOfVectors[arrayIndex][0].push_back(vectorOfVec3Arrays[vectorIndex][arrayIndex].x);
            arrayOfArrayOfVectors[arrayIndex][1].push_back(vectorOfVec3Arrays[vectorIndex][arrayIndex].y);
            arrayOfArrayOfVectors[arrayIndex][2].push_back(vectorOfVec3Arrays[vectorIndex][arrayIndex].z);
        }
    }

    return arrayOfArrayOfVectors;
}

/*
template <typename T, std::size_t N>
void saveTemporalVectorOfEnumsToHDF5(const std::vector<std::array<T, N>> & vectorOfEnumArrays, HighFive::File & file,
                                     int startOffset, const string & baseString) {
    std::array<std::vector<int>, N> arrayOfIntVectors = vectorOfEnumArraysToArrayOfIntVectors(vectorOfEnumArrays);
    for (size_t arrayIndex = 0; arrayIndex < arrayOfIntVectors.size(); arrayIndex++) {
        std::cout << "adding" << baseString + "(t" + toSignedIntString(arrayIndex + startOffset) + ")" << std::endl;
        H5Easy::dump(file, "/data/" +
                           baseString + "(t" + toSignedIntString(arrayIndex + startOffset, true) + ")",
                     arrayOfIntVectors[arrayIndex], QueryResult::defaultHDF5DumpOption);
    }
}

template <typename T, std::size_t N>
void saveTemporalArrayOfVectorsToHDF5(const std::vector<std::array<T, N>> & vectorOfArrays, HighFive::File & file,
                                      int startOffset, const string & baseString) {
    std::array<std::vector<T>, N> arrayOfVectors = vectorOfArraysToArrayOfVectors(vectorOfArrays);
    for (size_t arrayIndex = 0; arrayIndex < arrayOfVectors.size(); arrayIndex++) {
        H5Easy::dump(file, "/data/" +
                           baseString + "(t" + toSignedIntString(arrayIndex + startOffset, true) + ")",
                     arrayOfVectors[arrayIndex], QueryResult::defaultHDF5DumpOption);
    }
}
 */

template <std::size_t N>
void saveTemporalArrayOfVec2VectorsToHDF5(const std::vector<std::array<Vec2, N>> & vectorOfVec2Arrays, HighFive::File & file,
                                          int startOffset, const string & baseString,
                                          const HighFive::DataSetCreateProps & hdf5CreateProps) {
    std::array<std::array<std::vector<double>, 2>, N> arrayOfArrayOfVectors =
            vectorOfVec2ArraysToArrayOfArrayOfVectors(vectorOfVec2Arrays);
    for (size_t arrayIndex = 0; arrayIndex < arrayOfArrayOfVectors.size(); arrayIndex++) {
        file.createDataSet("/data/" +
                           baseString + " x (t" + toSignedIntString(arrayIndex + startOffset, true) + ")",
                           arrayOfArrayOfVectors[arrayIndex][0], hdf5CreateProps);
        file.createDataSet("/data/" +
                           baseString + " y (t" + toSignedIntString(arrayIndex + startOffset, true) + ")",
                           arrayOfArrayOfVectors[arrayIndex][1], hdf5CreateProps);
    }
}

/*
template <std::size_t N>
void saveTemporalArrayOfVec3VectorsToHDF5(const std::vector<std::array<Vec3, N>> & vectorOfVec3Arrays, HighFive::File & file,
                                          int startOffset, const string & baseString) {
    std::array<std::array<std::vector<double>, 3>, N> arrayOfArrayOfVectors =
            vectorOfVec3ArraysToArrayOfArrayOfVectors(vectorOfVec3Arrays);
    for (size_t arrayIndex = 0; arrayIndex < arrayOfArrayOfVectors.size(); arrayIndex++) {
        H5Easy::dump(file, "/data/" +
                           baseString + " x (t" + toSignedIntString(arrayIndex + startOffset, true) + ")",
                     arrayOfArrayOfVectors[arrayIndex][0], QueryResult::defaultHDF5DumpOption);
        H5Easy::dump(file, "/data/" +
                           baseString + " y (t" + toSignedIntString(arrayIndex + startOffset, true) + ")",
                     arrayOfArrayOfVectors[arrayIndex][1], QueryResult::defaultHDF5DumpOption);
        H5Easy::dump(file, "/data/" +
                           baseString + " z (t" + toSignedIntString(arrayIndex + startOffset, true) + ")",
                     arrayOfArrayOfVectors[arrayIndex][2], QueryResult::defaultHDF5DumpOption);
    }
}
 */

void mergeThreadResults(int numThreads, vector<RangeIndexEntry> &rowIndicesPerRound, const vector<vector<int64_t>> & tmpRoundIds,
                        const vector<vector<int64_t>> & tmpRoundStarts, const vector<vector<int64_t>> & tmpRoundSizes,
                        vector<int64_t> & resultStartTickId, int64_t & resultSize,
                        const std::function<void(int64_t, int64_t)> & appendToResult);

/*
class NoSourceTargetQuery : public QueryResult {
public:
    string toCSVFiltered(const Position & position, string game) {
        stringstream ss;
        ss << "demo tick,demo file";
        for (const auto & extraColName : getExtraColumnNames()) {
            ss << "," << extraColName;
        }
        ss << std::endl;
        for (int64_t i = 0; i < positionIndex.size(); i++) {
            int64_t posIdx = positionIndex[i];
            string curGame = position.fileNames[position.demoFile[posIdx]];
            if (curGame.compare(game) == 0 || game == "") {
                ss << position.demoTickNumber[posIdx] << "," << position.fileNames[position.demoFile[posIdx]];
                for (const auto & extraColValue : getExtraRow(position, i, posIdx)) {
                    ss << "," << extraColValue;
                }
                ss << std::endl;
            }
        }
        return ss.str();
    }

    string toCSV(const Position & position) {
        stringstream ss;
        ss << "demo tick,demo file";
        for (const auto & extraColName : getExtraColumnNames()) {
            ss << "," << extraColName;
        }
        ss << std::endl;
        for (int64_t i = 0; i < positionIndex.size(); i++) {
            int64_t posIdx = positionIndex[i];
            string curGame = position.fileNames[position.demoFile[posIdx]];
            ss << position.demoTickNumber[posIdx] << "," << position.fileNames[position.demoFile[posIdx]];
            for (const auto & extraColValue : getExtraRow(position, i, posIdx)) {
                ss << "," << extraColValue;
            }
            ss << std::endl;
        }
        return ss.str();
    };

    vector<string> getKeyNames() {
        return {};
    }

    DataType getDatatype() {
        return noSrcTarget;
    }
};

class JustTargetResult : public QueryResult {
public:
    vector<string> targetNames;

    vector<vector<int>> targets;

    string toCSVFiltered(const Position & position, string game) {
        stringstream ss;
        ss << "demo tick,demo file";
        for (const auto & targetName : targetNames) {
            ss << "," << targetName;
        }
        for (const auto & extraColName : getExtraColumnNames()) {
            ss << "," << extraColName;
        }
        ss << std::endl;
        for (int64_t i = 0; i < positionIndex.size(); i++) {
            int64_t posIdx = positionIndex[i];
            string curGame = position.fileNames[position.demoFile[posIdx]];
            if (curGame.compare(game) == 0 || game == "") {
                ss << position.demoTickNumber[posIdx] << "," << position.fileNames[position.demoFile[posIdx]];
                for (int j = 0; j < targets[i].size(); j++) {
                    ss << "," << position.players[targets[i][j]].name[posIdx];
                }
                for (const auto & extraColValue : getExtraRow(position, i, posIdx)) {
                    ss << "," << extraColValue;
                }
                ss << std::endl;
            }
        }
        return ss.str();
    }

    string toCSV(const Position & position) {
        stringstream ss;
        ss << "demo tick,demo file";
        for (const auto & targetName : targetNames) {
            ss << "," << targetName;
        }
        for (const auto & extraColName : getExtraColumnNames()) {
            ss << "," << extraColName;
        }
        ss << std::endl;
        for (int64_t i = 0; i < positionIndex.size(); i++) {
            int64_t posIdx = positionIndex[i];
            ss << position.demoTickNumber[posIdx] << "," << position.fileNames[position.demoFile[posIdx]];
            for (int j = 0; j < targets[i].size(); j++) {
                ss << "," << position.players[targets[i][j]].name[posIdx];
            }
            for (const auto & extraColValue : getExtraRow(position, i, posIdx)) {
                ss << "," << extraColValue;
            }
            ss << std::endl;
        }
        return ss.str();
    };

    vector<string> getKeyNames() {
        return targetNames;
    }

    DataType getDatatype() {
        return justTarget;
    }

};

class SourceAndTargetResult : public QueryResult {
public:
    string sourceName;
    vector<string> targetNames;

    vector<int> sources;
    vector<vector<int>> targets;

    string toCSVFiltered(const Position & position, string game) {
        stringstream ss;
        ss << "demo tick,demo file," << sourceName;
        for (const auto & targetName : targetNames) {
            ss << "," << targetName;
        }
        for (const auto & extraColName : getExtraColumnNames()) {
            ss << "," << extraColName;
        }
        ss << std::endl;
        for (int64_t i = 0; i < positionIndex.size(); i++) {
            int64_t posIdx = positionIndex[i];
            string curGame = position.fileNames[position.demoFile[posIdx]];
            if (curGame.compare(game) == 0 || game == "") {
                ss << position.demoTickNumber[posIdx] << "," << position.fileNames[position.demoFile[posIdx]] << ","
                   << position.players[sources[i]].name[posIdx];
                for (int j = 0; j < targets[i].size(); j++) {
                    ss << "," << position.players[targets[i][j]].name[posIdx];
                }
                for (const auto & extraColValue : getExtraRow(position, i, posIdx)) {
                    ss << "," << extraColValue;
                }
                ss << std::endl;
            }
        }
        return ss.str();
    }

    string toCSV(const Position & position) {
        stringstream ss;
        ss << "demo tick,demo file," << sourceName;
        for (const auto & targetName : targetNames) {
            ss << "," << targetName;
        }
        for (const auto & extraColName : getExtraColumnNames()) {
            ss << "," << extraColName;
        }
        ss << std::endl;
        for (int64_t i = 0; i < positionIndex.size(); i++) {
            int64_t posIdx = positionIndex[i];
            ss << position.demoTickNumber[posIdx] << "," << position.fileNames[position.demoFile[posIdx]] << ","
               << position.players[sources[i]].name[posIdx];
            for (int j = 0; j < targets[i].size(); j++) {
                ss << "," << position.players[targets[i][j]].name[posIdx];
            }
            for (const auto & extraColValue : getExtraRow(position, i, posIdx)) {
                ss << "," << extraColValue;
            }
            ss << std::endl;
        }
        return ss.str();
    };

    vector<string> getKeyNames() {
        vector<string> result = {sourceName};
        result.insert(result.end(), targetNames.begin(), targetNames.end());
        return result;
    }

    DataType getDatatype() {
        return srcAndTarget;
    }

};

struct SourceAndTarget {
    int source, target;
    bool operator <(const SourceAndTarget& cv) const {
        return source < cv.source || ((source == cv.source) && target < cv.target);
    }
};
*/
#endif //CSKNOW_QUERY_H
