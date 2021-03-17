#ifndef CSKNOW_WALLERS_H
#define CSKNOW_WALLERS_H
#include "load_data.h"
#include "query.h"
#include <string>
using std::string;

class WallersResult : public SingleSourceSingleTargetResult {
public:
    vector<int> & cheaters = sources;
    vector<int> & victims = targets;

    WallersResult() {
        sourceName = "waller";
        targetName = "victim";
    }
};

WallersResult queryWallers(const Position & position, const Spotted & spotted);

#endif //CSKNOW_WALLERS_H
