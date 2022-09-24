#ifndef CSKNOW_QUERY_H
#define CSKNOW_QUERY_H
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <functional>
#include "load_data.h"
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
    bool overlayLabels = false;
    bool havePlayerLabels = false;
    // these are offsets relative to other columns start
    int playersToLabelColumn = 0;
    int playerLabelIndicesColumn = 0;
    vector<string> playerLabels;
    bool havePerTickAimTable = false;
    string perTickAimTable;
    int eventIdColumn = -1;
    string extension = ".query";
    vector<int> keyPlayerColumns = {};
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

    string toCSV(int64_t otherTableIndex) {
        std::stringstream ss;
        addHeader(ss);
        vector<int64_t> filter = filterByForeignKey(otherTableIndex);
        for (const auto & index : filter) {
            oneLineToCSV(index, ss);
        }
        return ss.str();
    }

    void addHeader(stringstream & ss) {
        ss << "id";
        for (const auto & foreignKey : getForeignKeyNames()) {
            ss << "," << foreignKey;
        }
        for (const auto & otherCol : getOtherColumnNames()) {
            ss << "," << otherCol;
        }
        ss << std::endl;
    }

    static
    void commaSeparateList(stringstream & ss, vector<string> list, const string& separator = ",") {
        if (list.empty()) {
            return;
        }
        ss << list[0];
        for (size_t i = 1; i < list.size(); i++) {
            ss << separator << list[i];
        }
    }

    // find all rows with foreign key that reference another table
    virtual vector<int64_t> filterByForeignKey(int64_t otherTableIndex) = 0;
    virtual void oneLineToCSV(int64_t index, stringstream & ss) = 0;
    virtual vector<string> getForeignKeyNames() = 0;
    virtual vector<string> getOtherColumnNames() = 0;
};

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
