#ifndef CSKNOW_QUERY_H
#define CSKNOW_QUERY_H
#include <vector>
#include <sstream>
using std::vector;
using std::stringstream;


class QueryResult {
public:
    vector<int64_t> positionIndex;
    vector<string> demoFile;

    virtual string toCSVFiltered(const Position & position, string game) = 0;
    virtual string toCSV(const Position & position) = 0;
};

class SingleSourceSingleTargetResult : public QueryResult {
public:
    string sourceName;
    string targetName;

    vector<int> sources;
    vector<int> targets;

    virtual string getExtraColumns() = 0;
    virtual string getExtraRow(const Position & position, int64_t index) = 0;

    string toCSVFiltered(const Position & position, string game) {
        stringstream ss;
        ss << "demo tick,demo file," << sourceName << "," << targetName << std::endl;
        for (int64_t i = 0; i < positionIndex.size(); i++) {
            int64_t posIdx = positionIndex[i];
            string curGame = position.fileNames[position.demoFile[posIdx]];
            if (curGame.compare(game) == 0 || game == "") {
                ss << position.demoTickNumber[posIdx] << "," << position.fileNames[position.demoFile[posIdx]] << ","
                   << position.players[sources[i]].name[posIdx] << "," << position.players[targets[i]].name[posIdx] << std::endl;
            }
        }
        return ss.str();
    }

    string toCSV(const Position & position) {
        stringstream ss;
        ss << "demo tick,demo file," << sourceName << "," << targetName << std::endl;
        for (int64_t i = 0; i < positionIndex.size(); i++) {
            int64_t posIdx = positionIndex[i];
            ss << position.demoTickNumber[posIdx] << "," << position.fileNames[position.demoFile[posIdx]] << ","
               << position.players[sources[i]].name[posIdx] << "," << position.players[targets[i]].name[posIdx] << std::endl;
        }
        return ss.str();
    };

};

struct SourceAndTarget {
    int source, target;
    bool operator <(const SourceAndTarget& cv) const {
        return source < cv.source || ((source == cv.source) && target < cv.target);
    }
};

#endif //CSKNOW_QUERY_H
