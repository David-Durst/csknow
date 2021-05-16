//
// Created by durst on 5/16/21.
//

#ifndef CSKNOW_BASE_TABLES_H
#define CSKNOW_BASE_TABLES_H
#include "query.h"
#include "load_data.h"

class QueryGames : QueryResult {
    const Games & games;
    QueryGames(Games & games) : games(games), size(games.size) { }

    string oneLineToCSV(int64_t index, stringstream & ss) {
        ss << games.id[index] << "," << games.demoFile[index] << ","
           << games.demoTickRate[index] << "," << games.gameTickRate[index];
        ss << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {};
    }

    vector<string> getOtherColumnNames() {
        return {"demo file", "demo tick rate", "game tick rate"}
    }
};


#endif //CSKNOW_BASE_TABLES_H
