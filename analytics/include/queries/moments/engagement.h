//
// Created by durst on 7/6/22.
//

#ifndef CSKNOW_ENGAGEMENT_H
#define CSKNOW_ENGAGEMENT_H
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "load_data.h"
#include "queries/query.h"
#include "geometry.h"
#include "enum_helpers.h"
#define PRE_ENGAGEMENT_SECONDS 1.0
#define POST_ENGAGEMENT_SECONDS 0.2
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::map;

enum class EngagementRole {
    Attacker,
    Victim,
    NUM_ENGAGEMENT_ROLES [[maybe_unused]]
};

class EngagementResult : public QueryResult {
public:
    vector<RangeIndexEntry> rowIndicesPerRound;
    vector<int64_t> startTickId;
    vector<int64_t> endTickId;
    vector<int64_t> tickLength;
    vector<vector<int64_t>> playerId;
    vector<vector<EngagementRole>> role;
    vector<vector<int64_t>> hurtTickIds;
    vector<vector<int64_t>> hurtIds;
    IntervalIndex engagementsPerTick;


    EngagementResult() {
        variableLength = true;
        startTickColumn = 0;
        perEventLengthColumn = 2;
        havePlayerLabels = true;
        playerLabels = {"A", "T"};
        playersToLabelColumn = 0;
        playerLabelIndicesColumn = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        vector<int64_t> result;
        for (int64_t i = rowIndicesPerRound[otherTableIndex].minId; i <= rowIndicesPerRound[otherTableIndex].maxId; i++) {
            if (i == -1) {
                continue;
            }
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << index << "," << startTickId[index] << "," << endTickId[index] << "," << tickLength[index] << ",";

        vector<string> tmp;
        for (int64_t pId : playerId[index]) {
            tmp.push_back(std::to_string(pId));
        }
        commaSeparateList(s, tmp, ";");
        s << ",";

        tmp.clear();
        for (EngagementRole r : role[index]) {
            tmp.push_back(std::to_string(enumAsInt(r)));
        }
        commaSeparateList(s, tmp, ";");
        s << ",";

        tmp.clear();
        for (int64_t hId : hurtTickIds[index]) {
            tmp.push_back(std::to_string(hId));
        }
        commaSeparateList(s, tmp, ";");
        s << ",";

        tmp.clear();
        for (int64_t hId : hurtIds[index]) {
            tmp.push_back(std::to_string(hId));
        }
        commaSeparateList(s, tmp, ";");

        s << std::endl;
    }

    [[nodiscard]]
    vector<string> getForeignKeyNames() override {
        return {"start tick id", "end tick id", "length"};
    }

    [[nodiscard]]
    vector<string> getOtherColumnNames() override {
        return {"player ids", "roles", "hurt tick ids", "hurt ids"};
    }

    void toHDF5Inner(HighFive::File & file) override {

        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        hdf5FlatCreateProps.add(HighFive::Deflate(6));
        hdf5FlatCreateProps.add(HighFive::Chunking(startTickId.size()));

        file.createDataSet("/data/start tick id", startTickId, hdf5FlatCreateProps);
        file.createDataSet("/data/end tick id", endTickId, hdf5FlatCreateProps);
        file.createDataSet("/data/tick length", tickLength, hdf5FlatCreateProps);
        file.createDataSet("/data/attacker id", vectorOfVectorToVectorSelector(playerId, 0), hdf5FlatCreateProps);
        file.createDataSet("/data/attacker role",
                           vectorOfEnumsToVectorOfInts(vectorOfVectorToVectorSelector(role, 0)),
                           hdf5FlatCreateProps);
        file.createDataSet("/data/target id", vectorOfVectorToVectorSelector(playerId, 1), hdf5FlatCreateProps);
        file.createDataSet("/data/target role",
                           vectorOfEnumsToVectorOfInts(vectorOfVectorToVectorSelector(role, 1)),
                           hdf5FlatCreateProps);
    }
};


EngagementResult queryEngagementResult(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                       const Hurt & hurt);

#endif //CSKNOW_ENGAGEMENT_H
