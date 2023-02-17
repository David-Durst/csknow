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
    virtual void toHDF5Inner(HighFive::File &) {
        throw std::runtime_error("HDFS saving not implemented for this query yet");
    }

    void toHDF5(const string& filePath);

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
std::vector<array<int, N>> vectorOfEnumArraysToVectorOfIntArrays(const std::vector<std::array<T, N>> & vectorOfArrays) {
    std::vector<std::array<int, N>> result;
    result.reserve(vectorOfArrays.size());

    for (size_t vectorIndex = 0; vectorIndex < vectorOfArrays.size(); vectorIndex++) {
        result.push_back({});
        for (size_t arrayIndex = 0; arrayIndex < vectorOfArrays.front().size(); arrayIndex++) {
            result[vectorIndex][arrayIndex] = enumAsInt(vectorOfArrays[vectorIndex][arrayIndex]);
        }
    }

    return result;
}

template <std::size_t N>
std::array<std::vector<std::array<float, N>>, 2>
vectorOfVec2ArraysToArrayOfFloatVectors(const std::vector<std::array<Vec2, N>> & vectorOfVec2Arrays){
    std::array<std::vector<std::array<float, N>>, 2> result;
    for (size_t i = 0; i < result.size(); i++) {
        result[0].reserve(vectorOfVec2Arrays.size());
        result[1].reserve(vectorOfVec2Arrays.size());
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfVec2Arrays.size(); vectorIndex++) {
        result[0].push_back({});
        result[1].push_back({});
        for (size_t arrayIndex = 0; arrayIndex < vectorOfVec2Arrays.front().size(); arrayIndex++) {
            result[0][vectorIndex][arrayIndex] =
                static_cast<float>(vectorOfVec2Arrays[vectorIndex][arrayIndex].x);
            result[1][vectorIndex][arrayIndex] =
                static_cast<float>(vectorOfVec2Arrays[vectorIndex][arrayIndex].y);
        }
    }

    return result;
}

template <std::size_t N>
std::array<std::vector<std::array<float, N>>, 3>
vectorOfVec3ArraysToArrayOfFloatVectors(const std::vector<std::array<Vec3, N>> & vectorOfVec3Arrays){
    std::array<std::vector<std::array<float, N>>, 3> result;
    for (size_t i = 0; i < result.size(); i++) {
        result[0].reserve(vectorOfVec3Arrays.size());
        result[1].reserve(vectorOfVec3Arrays.size());
        result[2].reserve(vectorOfVec3Arrays.size());
    }

    for (size_t vectorIndex = 0; vectorIndex < vectorOfVec3Arrays.size(); vectorIndex++) {
        result[0].push_back({});
        result[1].push_back({});
        result[2].push_back({});
        for (size_t arrayIndex = 0; arrayIndex < vectorOfVec3Arrays.front().size(); arrayIndex++) {
            result[0][vectorIndex][arrayIndex] =
                static_cast<float>(vectorOfVec3Arrays[vectorIndex][arrayIndex].x);
            result[1][vectorIndex][arrayIndex] =
                static_cast<float>(vectorOfVec3Arrays[vectorIndex][arrayIndex].y);
            result[2][vectorIndex][arrayIndex] =
                static_cast<float>(vectorOfVec3Arrays[vectorIndex][arrayIndex].z);
        }
    }

    return result;
}

template <typename T, std::size_t N>
void saveTemporalVectorOfEnumsToHDF5(const std::vector<std::array<T, N>> & vectorOfEnumArrays, HighFive::File & file,
                                     int startOffset, const string & baseString,
                                     const HighFive::DataSetCreateProps & hdf5CreateProps) {
    std::vector<std::array<int, N>> vectorOfIntArrays = vectorOfEnumArraysToVectorOfIntArrays(vectorOfEnumArrays);
    HighFive::DataSet dataset = file.createDataSet("/data/" + baseString, vectorOfIntArrays, hdf5CreateProps);
    HighFive::Attribute names = dataset.createAttribute<std::string>("column names",
                                                                     HighFive::DataSpace::From(baseString));
    std::stringstream namesStream;
    for (size_t arrayIndex = 0; arrayIndex < vectorOfEnumArrays.front().size(); arrayIndex++) {
        if (arrayIndex > 0) {
            namesStream << ",";
        }
        namesStream << baseString + " (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
    }
    names.write(namesStream.str());
}

template <typename T, std::size_t N>
void saveTemporalArrayOfVectorsToHDF5(const std::vector<std::array<T, N>> & vectorOfArrays, HighFive::File & file,
                                      int startOffset, const string & baseString,
                                      const HighFive::DataSetCreateProps & hdf5CreateProps) {
    HighFive::DataSet dataset = file.createDataSet("/data/" + baseString, vectorOfArrays, hdf5CreateProps);
    HighFive::Attribute names = dataset.createAttribute<std::string>("column names",
                                                                     HighFive::DataSpace::From(baseString));
    std::stringstream namesStream;
    for (size_t arrayIndex = 0; arrayIndex < vectorOfArrays.front().size(); arrayIndex++) {
        if (arrayIndex > 0) {
            namesStream << ",";
        }
        namesStream << baseString + " (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
    }
    names.write(namesStream.str());
}

template <std::size_t N>
void saveTemporalArrayOfVec2VectorsToHDF5(const std::vector<std::array<Vec2, N>> & vectorOfVec2Arrays, HighFive::File & file,
                                          int startOffset, const string & baseString,
                                          const HighFive::DataSetCreateProps & hdf5CreateProps) {
    std::array<std::vector<std::array<float, N>>, 2> arrayOfArrayOfVectors =
        vectorOfVec2ArraysToArrayOfFloatVectors(vectorOfVec2Arrays);
    HighFive::DataSet xDataset =
        file.createDataSet("/data/" + baseString + " x", arrayOfArrayOfVectors[0], hdf5CreateProps);
    HighFive::DataSet yDataset =
        file.createDataSet("/data/" + baseString + " y", arrayOfArrayOfVectors[1], hdf5CreateProps);

    HighFive::Attribute xNames = xDataset.createAttribute<std::string>("column names",
                                                                         HighFive::DataSpace::From(baseString));
    HighFive::Attribute yNames = yDataset.createAttribute<std::string>("column names",
                                                                         HighFive::DataSpace::From(baseString));

    std::stringstream xNamesStream, yNamesStream;
    for (size_t arrayIndex = 0; arrayIndex < vectorOfVec2Arrays.front().size(); arrayIndex++) {
        if (arrayIndex > 0) {
            xNamesStream << ",";
            yNamesStream << ",";
        }
        xNamesStream << baseString + " x (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
        yNamesStream << baseString + " y (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
    }
    xNames.write(xNamesStream.str());
    yNames.write(yNamesStream.str());
}

template <std::size_t N>
void saveTemporalArrayOfVec3VectorsToHDF5(const std::vector<std::array<Vec3, N>> & vectorOfVec3Arrays, HighFive::File & file,
                                          int startOffset, const string & baseString,
                                          const HighFive::DataSetCreateProps & hdf5CreateProps) {
    std::array<std::vector<std::array<float, N>>, 3> arrayOfArrayOfVectors =
            vectorOfVec3ArraysToArrayOfFloatVectors(vectorOfVec3Arrays);
    HighFive::DataSet xDataset =
        file.createDataSet("/data/" + baseString + " x", arrayOfArrayOfVectors[0], hdf5CreateProps);
    HighFive::DataSet yDataset =
        file.createDataSet("/data/" + baseString + " y", arrayOfArrayOfVectors[1], hdf5CreateProps);
    HighFive::DataSet zDataset =
        file.createDataSet("/data/" + baseString + " z", arrayOfArrayOfVectors[2], hdf5CreateProps);

    HighFive::Attribute xNames = xDataset.createAttribute<std::string>("column names",
                                                                       HighFive::DataSpace::From(baseString));
    HighFive::Attribute yNames = yDataset.createAttribute<std::string>("column names",
                                                                       HighFive::DataSpace::From(baseString));
    HighFive::Attribute zNames = zDataset.createAttribute<std::string>("column names",
                                                                       HighFive::DataSpace::From(baseString));

    std::stringstream xNamesStream, yNamesStream, zNamesStream;
    for (size_t arrayIndex = 0; arrayIndex < vectorOfVec3Arrays.front().size(); arrayIndex++) {
        if (arrayIndex > 0) {
            xNamesStream << ",";
            yNamesStream << ",";
            zNamesStream << ",";
        }
        xNamesStream << baseString + " x (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
        yNamesStream << baseString + " y (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
        yNamesStream << baseString + " z (t" + toSignedIntString(arrayIndex + startOffset, true) + ")";
    }
    xNames.write(xNamesStream.str());
    yNames.write(yNamesStream.str());
    zNames.write(zNamesStream.str());
}

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
