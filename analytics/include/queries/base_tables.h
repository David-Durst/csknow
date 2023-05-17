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

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << games.id[index] << "," << games.demoFile[index] << ","
          << games.demoTickRate[index] << "," << games.gameTickRate[index];
        s << std::endl;
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

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << rounds.id[index] << "," << rounds.gameId[index] << "," << rounds.startTick[index]
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

    void toHDF5Inner(HighFive::File & file) override {
        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        hdf5FlatCreateProps.add(HighFive::Deflate(6));
        hdf5FlatCreateProps.add(HighFive::Chunking(rounds.id.size()));

        file.createDataSet("/data/game id", rounds.gameId, hdf5FlatCreateProps);
        file.createDataSet("/data/start tick", rounds.startTick, hdf5FlatCreateProps);
        file.createDataSet("/data/end tick", rounds.endTick, hdf5FlatCreateProps);
        file.createDataSet("/data/warmup", rounds.warmup, hdf5FlatCreateProps);
        file.createDataSet("/data/freeze time end", rounds.freezeTimeEnd, hdf5FlatCreateProps);
        file.createDataSet("/data/round number", rounds.roundNumber, hdf5FlatCreateProps);
        file.createDataSet("/data/round end reason", rounds.roundEndReason, hdf5FlatCreateProps);
        file.createDataSet("/data/winner", rounds.winner, hdf5FlatCreateProps);
        file.createDataSet("/data/tWins", rounds.tWins, hdf5FlatCreateProps);
        file.createDataSet("/data/ctWins", rounds.ctWins, hdf5FlatCreateProps);
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

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << players.id[index] << "," << players.gameId[index] << "," << players.name[index] << std::endl;
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

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << ticks.id[index] << "," << ticks.roundId[index] << "," << ticks.gameTime[index] << ","
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

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << pat.id[index] << "," << pat.tickId[index] << "," << pat.playerId[index] << ","
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

class QueryWeaponFire : public QueryResult {
public:
    const Rounds & rounds;
    const Ticks & ticks;
    const WeaponFire & weaponFire;
    QueryWeaponFire(const Rounds & rounds, const Ticks & ticks, const WeaponFire & weaponFire)
        : rounds(rounds), ticks(ticks), weaponFire(weaponFire) {
        this->size = weaponFire.size;
        this->startTickColumn = 0;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        vector<int64_t> result;
        for (int64_t i = rounds.ticksPerRound[otherTableIndex].minId;
             i <= rounds.ticksPerRound[otherTableIndex].maxId; i++) {
            for (const auto & [_0, _1, weaponFireIndex] :
                ticks.weaponFirePerTick.intervalToEvent.findOverlapping(i, i)) {
                result.push_back(weaponFireIndex);
            }
        }
        return result;
    }

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << weaponFire.id[index] << "," << weaponFire.tickId[index] << "," << weaponFire.shooter[index] << ","
          << weaponFire.weapon[index] << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"tick id", "shooter"};
    }

    vector<string> getOtherColumnNames() override {
        return {"weapon"};
    }

    void toHDF5Inner(HighFive::File & file) override {

        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        hdf5FlatCreateProps.add(HighFive::Deflate(6));
        hdf5FlatCreateProps.add(HighFive::Chunking(weaponFire.id.size()));

        file.createDataSet("/data/tick id", weaponFire.tickId, hdf5FlatCreateProps);
        file.createDataSet("/data/shooter", weaponFire.shooter, hdf5FlatCreateProps);
        file.createDataSet("/data/weapon", weaponFire.weapon, hdf5FlatCreateProps);
    }
};

class QueryHurt : public QueryResult {
public:
    const Rounds & rounds;
    const Ticks & ticks;
    const Hurt & hurt;
    QueryHurt(const Rounds & rounds, const Ticks & ticks, const Hurt & hurt)
        : rounds(rounds), ticks(ticks), hurt(hurt) {
        this->size = hurt.size;
        this->startTickColumn = 0;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        vector<int64_t> result;
        for (int64_t i = rounds.ticksPerRound[otherTableIndex].minId;
             i <= rounds.ticksPerRound[otherTableIndex].maxId; i++) {
            for (const auto & [_0, _1, killsIndex] :
                ticks.killsPerTick.intervalToEvent.findOverlapping(i, i)) {
                result.push_back(killsIndex);
            }
        }
        return result;
    }

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << hurt.id[index] << "," << hurt.tickId[index] << "," << hurt.attacker[index] << ","
          << hurt.victim[index] << "," << enumAsInt(hurt.weapon[index]) << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"tick id", "attacker", "victim"};
    }

    vector<string> getOtherColumnNames() override {
        return {"weapon"};
    }

    void toHDF5Inner(HighFive::File & file) override {

        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        hdf5FlatCreateProps.add(HighFive::Deflate(6));
        hdf5FlatCreateProps.add(HighFive::Chunking(hurt.id.size()));

        file.createDataSet("/data/tick id", hurt.tickId, hdf5FlatCreateProps);
        file.createDataSet("/data/attacker", hurt.attacker, hdf5FlatCreateProps);
        file.createDataSet("/data/victim", hurt.victim, hdf5FlatCreateProps);
        file.createDataSet("/data/weapon", vectorOfEnumsToVectorOfInts(hurt.weapon), hdf5FlatCreateProps);
    }
};

class QueryKills : public QueryResult {
public:
    const Rounds & rounds;
    const Ticks & ticks;
    const Kills & kills;
    QueryKills(const Rounds & rounds, const Ticks & ticks, const Kills & kills)
        : rounds(rounds), ticks(ticks), kills(kills) {
        this->size = kills.size;
        this->startTickColumn = 0;
        this->ticksPerEvent = 1;
    }

    vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
        vector<int64_t> result;
        for (int64_t i = rounds.ticksPerRound[otherTableIndex].minId;
             i <= rounds.ticksPerRound[otherTableIndex].maxId; i++) {
            for (const auto & [_0, _1, killsIndex] :
                ticks.killsPerTick.intervalToEvent.findOverlapping(i, i)) {
                result.push_back(killsIndex);
            }
        }
        return result;
    }

    void oneLineToCSV(int64_t index, std::ostream &s) override {
        s << kills.id[index] << "," << kills.tickId[index] << "," << kills.killer[index] << ","
          << kills.victim[index] << "," << kills.weapon[index] << std::endl;
    }

    vector<string> getForeignKeyNames() override {
        return {"tick id", "killer", "victim"};
    }

    vector<string> getOtherColumnNames() override {
        return {"weapon"};
    }

    void toHDF5Inner(HighFive::File & file) override {

        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        hdf5FlatCreateProps.add(HighFive::Deflate(6));
        hdf5FlatCreateProps.add(HighFive::Chunking(kills.id.size()));

        file.createDataSet("/data/tick id", kills.tickId, hdf5FlatCreateProps);
        file.createDataSet("/data/killer", kills.killer, hdf5FlatCreateProps);
        file.createDataSet("/data/victim", kills.victim, hdf5FlatCreateProps);
        file.createDataSet("/data/weapon", kills.weapon, hdf5FlatCreateProps);
    }
};

#endif //CSKNOW_BASE_TABLES_H
