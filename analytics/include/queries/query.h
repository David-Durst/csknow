#ifndef CSKNOW_QUERY_H
#define CSKNOW_QUERY_H
#include <vector>
#include <sstream>
using std::vector;
using std::stringstream;

enum DataType {
    noSrcTarget = 0,
    justSrc = 1,
    srcAndTarget = 2
};

class QueryResult {
public:
    vector<int64_t> positionIndex;
    vector<string> demoFile;

    virtual string toCSVFiltered(const Position & position, string game) = 0;
    virtual string toCSV(const Position & position) = 0;
    virtual string getSourceName() = 0;
    virtual vector<string> getTargetNames() = 0;
    virtual vector<string> getExtraColumnNames() = 0;
    virtual DataType getDatatype() = 0;
};

class SourceAndTargetResult : public QueryResult {
public:
    string sourceName;
    vector<string> targetNames;

    vector<int> sources;
    vector<vector<int>> targets;

    virtual vector<string> getExtraRow(const Position & position, int64_t index) = 0;

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
                for (const auto & extraColValue : getExtraRow(position, i)) {
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
            for (const auto & extraColValue : getExtraRow(position, i)) {
                ss << "," << extraColValue;
            }
            ss << std::endl;
        }
        return ss.str();
    };

    string getSourceName() {
        return sourceName;
    }

    vector<string> getTargetNames() {
        return targetNames;
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

#endif //CSKNOW_QUERY_H
