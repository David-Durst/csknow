#ifndef CSKNOW_BAITERS_H
#define CSKNOW_BAITERS_H
#include "load_data.h"
#include "query.h"
#include <string>
using std::string;

class BaitersResult : public SingleSourceSingleTargetResult {
public:
    vector<int> & baiters = sources;
    vector<int> & victims = targets;

    BaitersResult() {
        sourceName = "baiter";
        targetName = "victim";
    }
};

#endif //CSKNOW_BAITERS_H
