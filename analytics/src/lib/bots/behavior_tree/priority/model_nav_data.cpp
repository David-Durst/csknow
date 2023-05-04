//
// Created by durst on 5/3/23.
//

#include <iomanip>
#include "bots/behavior_tree/priority/model_nav_data.h"

struct OptionAndProb {
    string option;
    float prob;
};

string ModelNavData::print(const ServerState &) const {
    stringstream result;
    result << std::fixed << std::setprecision(2);

    vector<OptionAndProb> optionAndProbs;

    for (size_t i = 0; i < orderPlaceOptions.size(); i++) {
        optionAndProbs.push_back({orderPlaceOptions[i], orderPlaceProbs[i]});
    }
    std::sort(optionAndProbs.begin(), optionAndProbs.end(), [](const OptionAndProb & left, const OptionAndProb & right) {
        return left.prob > right.prob;
    });

    result << "cur place " << curPlace << ", next place " << nextPlace << ", next area " << nextArea
        << "order places: ";
    for (const auto & optionAndProb : optionAndProbs) {
        result << "(" << optionAndProb.option << "," << optionAndProb.prob << "), ";
    }

    return result.str();
}