#ifndef CSKNOW_WALLERS_H
#define CSKNOW_WALLERS_H
#include "load_data.h"
#include "query.h"

class WallersResult : public PredicateResult {
public:
    vector<int> cheaters;
    vector<int> victims;

    virtual string toCSV(const Position & position) {
        stringstream ss;
        ss << "demo tick,demo file,cheater,victim" << std::endl;
        for (int64_t i = 0; i < positionIndex.size(); i++) {
            ss << position.demoTickNumber[i] << "," << position.fileNames[position.demoFile[i]] << ","
                << cheaters[i] << "," << victims[i] << std::endl;
        }
        return ss.str();
    };
};

WallersResult queryWallers(const Position & position, const Spotted & spotted);

#endif //CSKNOW_WALLERS_H
