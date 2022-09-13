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
    QueryGames(Games & games) : games(games) {
        this->size = games.size;
        this->startTickColumn = -1;
        this->variableLength = false;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        return {};
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << games.id[index] << "," << games.demoFile[index] << ","
           << games.demoTickRate[index] << "," << games.gameTickRate[index];
        ss << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {};
    }

    vector<string> getOtherColumnNames() {
        return {"demo file", "demo tick rate", "game tick rate"};
    }
};

class QueryRounds : public QueryResult {
public:
    const Rounds & rounds;
    const Games & games;
    QueryRounds(const Games & games, const Rounds & rounds) : games(games), rounds(rounds) {
        this->size = rounds.size;
        this->startTickColumn = 1;
        this->variableLength = true;
        this->perEventLengthColumn = 3;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        vector<int64_t> result;
        for (int i = games.roundsPerGame[otherTableIndex].minId; i <= games.roundsPerGame[otherTableIndex].maxId; i++) {
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << rounds.id[index] << "," << rounds.gameId[index] << "," << rounds.startTick[index]
           << "," << rounds.endTick[index] << "," << rounds.endTick[index] - rounds.startTick[index] + 1 << ","
           << rounds.warmup[index] << "," << rounds.freezeTimeEnd[index] << "," << rounds.roundNumber[index] << ","
           << rounds.roundEndReason[index] << "," << rounds.winner[index] << ","
           << rounds.tWins[index] << "," << rounds.ctWins[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"game id", "start tick", "end tick", "round length"};
    }

    vector<string> getOtherColumnNames() {
        return {"warmup", "freeze time end", "round number", "round end reason", "winner",
                "t wins", "ct wins"};
    }
};

class QueryPlayers : public QueryResult {
public:
    const Players & players;
    const Games & games;
    QueryPlayers(const Games & games, const Players & players) : games(games), players(players) {
        this->size = players.size;
        this->startTickColumn = -1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        vector<int64_t> result;
        for (int i = games.playersPerGame[otherTableIndex].minId; i <= games.playersPerGame[otherTableIndex].maxId; i++) {
            result.push_back(i);
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << players.id[index] << "," << players.gameId[index] << "," << players.name[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"game id"};
    }

    vector<string> getOtherColumnNames() {
        return {"name"};
    }
};

class QueryTicks : public QueryResult {
public:
    const Ticks & ticks;
    const Rounds & rounds;
    QueryTicks(const Rounds & rounds, const Ticks & ticks) : rounds(rounds), ticks(ticks) {
        this->size = ticks.size;
        this->startTickColumn = -1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        vector<int64_t> result;
        for (int i = rounds.ticksPerRound[otherTableIndex].minId; i <= rounds.ticksPerRound[otherTableIndex].maxId; i++) {
            if (i != -1) {
                result.push_back(i);
            }
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << ticks.id[index] << "," << ticks.roundId[index] << "," << ticks.gameTime[index] << ","
           << ticks.demoTickNumber[index] << "," << ticks.gameTickNumber[index] << "," << ticks.bombCarrier[index] << ","
           << ticks.bombX[index] << "," << ticks.bombY[index] << "," << ticks.bombZ[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"round id"};
    }

    vector<string> getOtherColumnNames() {
        return {"game time", "demo tick number", "game tick number", "bomb carrier", "bomb x", "bomb y", "bomb z"};
    }
};

class QueryPlayerAtTick : public QueryResult {
public:
    const PlayerAtTick & pat;
    const Ticks & ticks;
    const Rounds & rounds;
    QueryPlayerAtTick(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & pat)
        : rounds(rounds), ticks(ticks), pat(pat) {
        this->size = pat.size;
        this->startTickColumn = 0;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) {
        vector<int64_t> result;
        for (int i = rounds.ticksPerRound[otherTableIndex].minId; i <= rounds.ticksPerRound[otherTableIndex].maxId; i++) {
            for (int j = ticks.patPerTick[i].minId; j <= ticks.patPerTick[i].maxId; j++) {
                if (j != -1) {
                    result.push_back(j);
                }
            }
        }
        return result;
    }

    void oneLineToCSV(int64_t index, stringstream & ss) {
        ss << pat.id[index] << "," << pat.tickId[index] << "," << pat.playerId[index] << ","
           << pat.posX[index] << "," << pat.posY[index] << "," << pat.posZ[index] << ","
           << pat.viewX[index] << "," << pat.viewY[index] << "," << pat.team[index] << ","
           << pat.health[index] << "," << pat.armor[index] << "," << pat.isAlive[index] << std::endl;
    }

    vector<string> getForeignKeyNames() {
        return {"tick id", "player id"};
    }

    vector<string> getOtherColumnNames() {
        return {"pos x", "pos y", "pos z", "view x", "view y", "team", "health", "armor", "is alive"};
    }
};

static
map<int64_t, int64_t> getPATIdForPlayerId(const Ticks & ticks, const PlayerAtTick & playerAtTick, int64_t tickIndex) {
    map<int64_t, int64_t> playerIdToPatID;
    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
         tickIndex <= ticks.patPerTick[tickIndex].maxId; tickIndex++) {
        playerIdToPatID[playerAtTick.playerId[patIndex]] = patIndex;
    }
    return playerIdToPatID;
}

#endif //CSKNOW_BASE_TABLES_H
