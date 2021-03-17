#ifndef CSKNOW_QUERY_H
#define CSKNOW_QUERY_H
#include <vector>
#include <sstream>
using std::vector;
using std::stringstream;


class QueryResult {
public:
    virtual string toCSVFiltered(const Position & position, string game) = 0;
    virtual string toCSV(const Position & position) = 0;
};

class PredicateResult : public QueryResult {
public:
    vector<int64_t> positionIndex;
    vector<string> demoFile;

    void collectResults(vector<int64_t> * tmpIndices, int numThreads) {
        for (int i = 0; i < numThreads; i++) {
            for (const auto & elem : tmpIndices[i]) {
                positionIndex.push_back(elem);
            }
        }
    }
};

class SingleSourceSingleTargetResult : public PredicateResult {
public:
    string sourceName;
    string targetName;

    vector<int> sources;
    vector<int> targets;

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

#endif //CSKNOW_QUERY_H
