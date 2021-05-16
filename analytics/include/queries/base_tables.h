//
// Created by durst on 5/16/21.
//

#ifndef CSKNOW_BASE_TABLES_H
#define CSKNOW_BASE_TABLES_H
#include "query.h"
#include "load_data.h"

class QueryGames : public QueryResult {
public:
    const Games & games;
    QueryGames(Games & games) : games(games) { this->size = size; }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        // no foreign keys in games, so no way to filter based on those foreign keys
        return {};
    }

    string oneLineToCSV(int64_t index, stringstream & ss) {
        ss << games.id[index] << "," << games.demoFile[index] << ","
           << games.demoTickRate[index] << "," << games.gameTickRate[index];
        ss << std::endl;
        return ss.str();
    }

    vector<string> getForeignKeyNames() {
        return {};
    }

    vector<string> getOtherColumnNames() {
        return {"demo file", "demo tick rate", "game tick rate"};
    }
};


#endif //CSKNOW_BASE_TABLES_H
