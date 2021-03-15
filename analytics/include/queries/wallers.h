#ifndef CSKNOW_WALLERS_H
#define CSKNOW_WALLERS_H
#include "load_data.h"
#include "query.h"
#include <string>
using std::string;

class WallersResult : public PredicateResult {
public:
    vector<int> cheaters;
    vector<int> victims;

    string toCSVFiltered(const Position & position, string game) {
        stringstream ss;
        ss << "demo tick,demo file,cheater,victim" << std::endl;
        for (int64_t i = 0; i < positionIndex.size(); i++) {
            int64_t posIdx = positionIndex[i];
            string curGame = position.fileNames[position.demoFile[posIdx]];
            if (curGame.compare(game) == 0 || game == "") {
                ss << position.demoTickNumber[posIdx] << "," << position.fileNames[position.demoFile[posIdx]] << ","
                   << position.players[cheaters[i]].name[posIdx] << "," << position.players[victims[i]].name[posIdx] << std::endl;
            }
        }
        return ss.str();
    }

    virtual string toCSV(const Position & position) {
        stringstream ss;
        ss << "demo tick,demo file,cheater,victim" << std::endl;
        for (int64_t i = 0; i < positionIndex.size(); i++) {
            int64_t posIdx = positionIndex[i];
            ss << position.demoTickNumber[posIdx] << "," << position.fileNames[position.demoFile[posIdx]] << ","
                << position.players[cheaters[i]].name[posIdx] << "," << position.players[victims[i]].name[posIdx] << std::endl;
        }
        return ss.str();
    };
};

WallersResult queryWallers(const Position & position, const Spotted & spotted);

#endif //CSKNOW_WALLERS_H
