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
    explicit QueryGames(Games & games) : games(games) {
        this->size = games.size;
        this->startTickColumn = -1;
        this->variableLength = false;
    }

    vector<int64_t> filterByForeignKey(int64_t) override {
        return {};
    }

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << games.id[index] << "," << games.demoFile[index] << ","
           << games.demoTickRate[index] << "," << games.gameTickRate[index];
        ss << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {};
    }

    vector<string> getOtherColumnNames() override {
        return {"demo file", "demo tick rate", "game tick rate"};
    }
};

class QueryRounds : public QueryResult {
public:
    const Games & games;
    const Rounds & rounds;
    QueryRounds(const Games & games, const Rounds & rounds) : games(games), rounds(rounds) {
        this->size = rounds.size;
        this->startTickColumn = 1;
        this->variableLength = true;
        this->perEventLengthColumn = 3;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        vector<int64_t> result;
        for (int64_t i = games.roundsPerGame[otherTableIndex].minId;
            i <= games.roundsPerGame[otherTableIndex].maxId; i++) {
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << rounds.id[index] << "," << rounds.gameId[index] << "," << rounds.startTick[index]
           << "," << rounds.endTick[index] << "," << rounds.endTick[index] - rounds.startTick[index] + 1 << ","
           << rounds.warmup[index] << "," << rounds.freezeTimeEnd[index] << "," << rounds.roundNumber[index] << ","
           << rounds.roundEndReason[index] << "," << rounds.winner[index] << ","
           << rounds.tWins[index] << "," << rounds.ctWins[index] << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"game id", "start tick", "end tick", "round length"};
    }

    vector<string> getOtherColumnNames() override {
        return {"warmup", "freeze time end", "round number", "round end reason", "winner",
                "t wins", "ct wins"};
    }
};

class QueryPlayers : public QueryResult {
public:
    const Games & games;
    const Players & players;
    QueryPlayers(const Games & games, const Players & players) : games(games), players(players) {
        this->size = players.size;
        this->startTickColumn = -1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        vector<int64_t> result;
        for (int64_t i = games.playersPerGame[otherTableIndex].minId;
            i <= games.playersPerGame[otherTableIndex].maxId; i++) {
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << players.id[index] << "," << players.gameId[index] << "," << players.name[index] << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"game id"};
    }

    vector<string> getOtherColumnNames() override {
        return {"name"};
    }
};

class QueryTicks : public QueryResult {
public:
    const Rounds & rounds;
    const Ticks & ticks;
    QueryTicks(const Rounds & rounds, const Ticks & ticks) : rounds(rounds), ticks(ticks) {
        this->size = ticks.size;
        this->startTickColumn = -1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        vector<int64_t> result;
        for (int64_t i = rounds.ticksPerRound[otherTableIndex].minId;
            i <= rounds.ticksPerRound[otherTableIndex].maxId; i++) {
            if (i != -1) {
                result.push_back(i);
            }
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << ticks.id[index] << "," << ticks.roundId[index] << "," << ticks.gameTime[index] << ","
           << ticks.demoTickNumber[index] << "," << ticks.gameTickNumber[index] << "," << ticks.bombCarrier[index] << ","
           << ticks.bombX[index] << "," << ticks.bombY[index] << "," << ticks.bombZ[index] << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"round id"};
    }

    vector<string> getOtherColumnNames() override {
        return {"game time", "demo tick number", "game tick number", "bomb carrier", "bomb x", "bomb y", "bomb z"};
    }
};

class QueryPlayerAtTick : public QueryResult {
public:
    const Rounds & rounds;
    const Ticks & ticks;
    const PlayerAtTick & pat;
    QueryPlayerAtTick(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & pat)
        : rounds(rounds), ticks(ticks), pat(pat) {
        this->size = pat.size;
        this->startTickColumn = 0;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        vector<int64_t> result;
        for (int64_t i = rounds.ticksPerRound[otherTableIndex].minId;
            i <= rounds.ticksPerRound[otherTableIndex].maxId; i++) {
            for (int64_t j = ticks.patPerTick[i].minId; j <= ticks.patPerTick[i].maxId; j++) {
                if (j != -1) {
                    result.push_back(j);
                }
            }
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) override {
        ss << pat.id[index] << "," << pat.tickId[index] << "," << pat.playerId[index] << ","
           << pat.posX[index] << "," << pat.posY[index] << "," << pat.posZ[index] << ","
           << pat.viewX[index] << "," << pat.viewY[index] << "," << pat.team[index] << ","
           << pat.health[index] << "," << pat.armor[index] << "," << pat.isAlive[index] << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"tick id", "player id"};
    }

    vector<string> getOtherColumnNames() override {
        return {"pos x", "pos y", "pos z", "view x", "view y", "team", "health", "armor", "is alive"};
    }
};

#endif //CSKNOW_BASE_TABLES_H
