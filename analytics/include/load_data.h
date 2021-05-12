#ifndef CSKNOW_LOAD_DATA_H
#define CSKNOW_LOAD_DATA_H
#define NUM_PLAYERS 10
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <map>
using std::string;
using std::vector;
using std::set;
using std::map;

class ColStore {
public:
    bool beenInitialized = false;
    int64_t size;
    vector<int64_t> id;
    set<int64_t> skipRows;
    virtual void init(int64_t rows, int64_t numFiles, vector<int64_t> id) {
        beenInitialized = true;
        size = rows;
        this->id = id;
    }
};

#define CT_TEAM 0
#define T_TEAM 1
#define SPECTATOR 2
struct PlayerPosition {
    char ** name;
    int8_t * team;
    double * xPosition;
    double * yPosition;
    double * zPosition;
    double * xViewDirection;
    double * yViewDirection;
    bool * isAlive;
    bool * isBlinded;
};

class EquipmentTable : public ColStore {
public:
    char ** name;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        name = (char **) malloc(rows * sizeof(char*));
    }

    EquipmentTable() { };
    ~EquipmentTable() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(name[row]);
        }
        free(name);
    }

    EquipmentTable(const EquipmentTable& other) = delete;
    EquipmentTable& operator=(const EquipmentTable& other) = delete;
};

class GameTypesTable : public ColStore {
public:
    char ** tableType;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        tableType = (char **) malloc(rows * sizeof(char*));
    }

    GameTypesTable() { };
    ~GameTypesTable() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(tableType[row]);
        }
        free(tableType);
    }

    GameTypesTable(const GameTypesTable& other) = delete;
    GameTypesTable& operator=(const GameTypesTable& other) = delete;
};

class HitGroupsTable : public ColStore {
public:
    char ** groupName;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        groupName = (char **) malloc(rows * sizeof(char*));
    }

    HitGroupsTable() { };
    ~HitGroupsTable() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(groupName[row]);
        }
        free(groupName);
    }

    HitGroupsTable(const HitGroupsTable& other) = delete;
    HitGroupsTable& operator=(const HitGroupsTable& other) = delete;
};

class GamesTable : public ColStore {
public:
    char ** demoFile;
    double * demoTickRate;
    double * gameTickRate;
    int64_t * gameType;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        demoFile = (char **) malloc(rows * sizeof(char*));
        demoTickRate = (double *) malloc(rows * sizeof(double));
        gameTickRate = (double *) malloc(rows * sizeof(double));
        gameType = (int64_t *) malloc(rows * sizeof(int64_t));
    }

    GamesTable() { };
    ~GamesTable() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(demoFile[row]);
        }
        free(demoFile);

        free(demoTickRate);
        free(gameTickRate);
        free(gameType);
    }

    GamesTable(const GamesTable& other) = delete;
    GamesTable& operator=(const GamesTable& other) = delete;
};

class PlayersTable : public ColStore {
public:
    int64_t * gameId;
    char ** name;
    int64_t * steamId;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        gameId = (int64_t *) malloc(rows * sizeof(int64_t));
        name = (char **) malloc(rows * sizeof(char*));
        steamId = (int64_t *) malloc(rows * sizeof(int64_t));
    }

    PlayersTable() { };
    ~PlayersTable() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(name[row]);
        }
        free(name);

        free(gameId);
        free(steamId);
    }

    PlayersTable(const PlayersTable& other) = delete;
    PlayersTable& operator=(const PlayersTable& other) = delete;
};

class RoundsTable : public ColStore {
public:
    int64_t * gameId;
    int64_t * startTick;
    int64_t * endTick;
    bool * warmup;
    int64_t * freezeTimeEnd;
    int8_t * roundNumber;
    int8_t * roundEndReason;
    int8_t * winner;
    int8_t * tWins;
    int8_t * ctWins;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        gameId = (int64_t *) malloc(rows * sizeof(int64_t));
        startTick = (int64_t *) malloc(rows * sizeof(int64_t));
        endTick = (int64_t *) malloc(rows * sizeof(int64_t));
        warmup = (bool *) malloc(rows * sizeof(bool));
        freezeTimeEnd = (int64_t *) malloc(rows * sizeof(int64_t));
        roundNumber = (int8_t *) malloc(rows * sizeof(int8_t));
        roundEndReason = (int8_t *) malloc(rows * sizeof(int8_t));
        winner = (int8_t *) malloc(rows * sizeof(int8_t));
        tWins = (int8_t *) malloc(rows * sizeof(int8_t));
        ctWins = (int8_t *) malloc(rows * sizeof(int8_t));
    }

    RoundsTable() { };
    ~RoundsTable() {
        if (!beenInitialized){
            return;
        }

        free(gameId);
        free(startTick);
        free(endTick);
        free(warmup);
        free(freezeTimeEnd);
        free(roundNumber);
        free(roundEndReason);
        free(winner);
        free(tWins);
        free(ctWins);
    }

    RoundsTable(const RoundsTable& other) = delete;
    RoundsTable& operator=(const RoundsTable& other) = delete;
};

class TicksTable: public ColStore {
public:
    int64_t * roundId;
    int64_t * gameTime;
    int64_t * demoTickNumber;
    int64_t * gameTickNumber;
    int64_t * bombCarrier;
    double * bombX;
    double * bombY;
    double * bombZ;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        roundId = (int64_t *) malloc(rows * sizeof(int64_t));
        gameTime = (int64_t *) malloc(rows * sizeof(int64_t));
        demoTickNumber = (int64_t *) malloc(rows * sizeof(int64_t));
        gameTickNumber = (int64_t *) malloc(rows * sizeof(int64_t));
        bombCarrier = (int64_t *) malloc(rows * sizeof(int64_t));
        bombX = (double *) malloc(rows * sizeof(double));
        bombY = (double *) malloc(rows * sizeof(double));
        bombZ = (double *) malloc(rows * sizeof(double));
    }

    TicksTable() { };
    ~TicksTable() {
        if (!beenInitialized){
            return;
        }
        free(roundId);
        free(gameTime);
        free(demoTickNumber);
        free(gameTickNumber);
        free(bombCarrier);
        free(bombX);
        free(bombY);
        free(bombZ);
    }

    TicksTable(const TicksTable& other) = delete;
    TicksTable& operator=(const TicksTable& other) = delete;

    /*
    // since spotted tracks names for spotted player, need to map that to the player index

    map<string, int> getPlayerNameToIndex(int64_t gameIndex) const {
        map<string, int> result;
        for (int i = 0; i < NUM_PLAYERS; i++) {
            result.insert({players[i].name[firstRowAfterWarmup[gameIndex]], i});
        }
        return result;
    }

    map<int, vector<int>> getEnemiesForTeam(int64_t gameIndex) const {
        map<int, vector<int>> result;
        result.insert({2, {}});
        result.insert({3, {}});
        for (int i = 0; i < NUM_PLAYERS; i++) {
            if (players[i].team[firstRowAfterWarmup[gameIndex]] == 2) {
                result[3].push_back(i);
            }
            else {
                result[2].push_back(i);
            }
        }
        return result;
    }
     */
};

class PlayerAtTickTable: public ColStore {
public:
    int64_t * playerId;
    int64_t * tickId;
    double * posX;
    double * posY;
    double * posZ;
    double * viewX;
    double * viewY;
    int8_t * team;
    double * health;
    double * armor;
    bool * isAlive;
    bool * isCrouching;
    bool * isAirborne;
    double * remainingFlashTime;
    int8_t * activeWeapon;
    int8_t * primaryWeapon;
    int8_t * primaryBulletsClip;
    int8_t * primaryBulletsReserve;
    int8_t * secondaryWeapon;
    int8_t * secondaryBulletsClip;
    int8_t * secondaryBulletsReserve;
    int8_t * numHe;
    int8_t * numFlash;
    int8_t * numSmoke;
    int8_t * numMolotov;
    int8_t * numIncendiary;
    int8_t * numDecoy;
    int8_t * numZeus;
    bool * hasDefuser;
    bool * hasBomb;
    int32_t * money;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        playerId = (int64_t *) malloc(rows * sizeof(int64_t));
        tickId = (int64_t *) malloc(rows * sizeof(int64_t));
        posX = (double *) malloc(rows * sizeof(double));
        posY = (double *) malloc(rows * sizeof(double));
        posZ = (double *) malloc(rows * sizeof(double));
        viewX = (double *) malloc(rows * sizeof(double));
        viewY = (double *) malloc(rows * sizeof(double));
        team = (int8_t *) malloc(rows * sizeof(int8_t));
        health = (double *) malloc(rows * sizeof(double));
        armor = (double *) malloc(rows * sizeof(double));
        isAlive = (bool *) malloc(rows * sizeof(bool));
        isCrouching = (bool *) malloc(rows * sizeof(bool));
        isAirborne = (bool *) malloc(rows * sizeof(bool));
        remainingFlashTime = (double *) malloc(rows * sizeof(double));
        activeWeapon = (int8_t *) malloc(rows * sizeof(int8_t));
        primaryWeapon = (int8_t *) malloc(rows * sizeof(int8_t));
        primaryBulletsClip = (int8_t *) malloc(rows * sizeof(int8_t));
        primaryBulletsReserve = (int8_t *) malloc(rows * sizeof(int8_t));
        secondaryWeapon = (int8_t *) malloc(rows * sizeof(int8_t));
        secondaryBulletsClip = (int8_t *) malloc(rows * sizeof(int8_t));
        secondaryBulletsReserve = (int8_t *) malloc(rows * sizeof(int8_t));
        numHe = (int8_t *) malloc(rows * sizeof(int8_t));
        numFlash = (int8_t *) malloc(rows * sizeof(int8_t));
        numSmoke = (int8_t *) malloc(rows * sizeof(int8_t));
        numMolotov = (int8_t *) malloc(rows * sizeof(int8_t));
        numIncendiary = (int8_t *) malloc(rows * sizeof(int8_t));
        numDecoy = (int8_t *) malloc(rows * sizeof(int8_t));
        numZeus = (int8_t *) malloc(rows * sizeof(int8_t));
        hasDefuser = (bool *) malloc(rows * sizeof(bool));
        hasBomb = (bool *) malloc(rows * sizeof(bool));
        money = (int32_t *) malloc(rows * sizeof(int32_t));
    }

    void makePitchNeg90To90() {
        for (int64_t i = 0; i < size; i++) {
            if (viewY[i] > 260.0) {
                viewY[i] -= 360;
            }
        }
    }


    PlayerAtTickTable() { };
    ~PlayerAtTickTable() {
        if (!beenInitialized){
            return;
        }
        free(playerId);
        free(tickId);
        free(posX);
        free(posY);
        free(posZ);
        free(viewX);
        free(team);
        free(health);
        free(armor);
        free(isAlive);
        free(isCrouching);
        free(isAirborne);
        free(remainingFlashTime);
        free(activeWeapon);
        free(primaryWeapon);
        free(primaryBulletsClip);
        free(primaryBulletsReserve);
        free(secondaryWeapon);
        free(secondaryBulletsClip);
        free(secondaryBulletsReserve);
        free(numHe);
        free(numFlash);
        free(numSmoke);
        free(numMolotov);
        free(numIncendiary);
        free(numDecoy);
        free(numZeus);
        free(hasDefuser);
        free(hasBomb);
        free(money);
    }

    PlayerAtTickTable(const PlayerAtTickTable& other) = delete;
    PlayerAtTickTable& operator=(const PlayerAtTickTable& other) = delete;

    /*
    // since spotted tracks names for spotted player, need to map that to the player index

    map<string, int> getPlayerNameToIndex(int64_t gameIndex) const {
        map<string, int> result;
        for (int i = 0; i < NUM_PLAYERS; i++) {
            result.insert({players[i].name[firstRowAfterWarmup[gameIndex]], i});
        }
        return result;
    }

    map<int, vector<int>> getEnemiesForTeam(int64_t gameIndex) const {
        map<int, vector<int>> result;
        result.insert({2, {}});
        result.insert({3, {}});
        for (int i = 0; i < NUM_PLAYERS; i++) {
            if (players[i].team[firstRowAfterWarmup[gameIndex]] == 2) {
                result[3].push_back(i);
            }
            else {
                result[2].push_back(i);
            }
        }
        return result;
    }
     */
};

class SpottedTable : public ColStore {
public:
    int64_t * tickId;
    int64_t * spottedPlayer;
    int64_t * spotterPlayer;
    bool * isSpotted;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        tickId = (int64_t *) malloc(rows * sizeof(int64_t));
        spottedPlayer = (int64_t *) malloc(rows * sizeof(int64_t));
        spotterPlayer = (int64_t *) malloc(rows * sizeof(int64_t));
        isSpotted = (bool *) malloc(rows * sizeof(bool));
    }

    SpottedTable() { };
    ~SpottedTable() {
        if (!beenInitialized){
            return;
        }
        free(tickId);
        free(spottedPlayer);
        free(spotterPlayer);
        free(isSpotted);
    }

    SpottedTable(const SpottedTable& other) = delete;
    SpottedTable& operator=(const SpottedTable& other) = delete;
};

class WeaponFireTable : public ColStore {
public:
    int64_t * tickId;
    int64_t * shooter;
    int8_t * weapon;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        tickId = (int64_t *) malloc(rows * sizeof(int64_t));
        shooter = (int64_t *) malloc(rows * sizeof(int64_t));
        weapon = (int8_t *) malloc(rows * sizeof(int8_t));
    }

    WeaponFireTable() { };
    ~WeaponFireTable() {
        if (!beenInitialized){
            return;
        }
        free(tickId);
        free(shooter);
        free(weapon);
    }

    WeaponFireTable(const WeaponFireTable& other) = delete;
    WeaponFireTable& operator=(const WeaponFireTable& other) = delete;
};

class PlayerHurtTable : public ColStore {
public:
    char ** victimName;
    int32_t * armorDamage;
    int32_t * armor;
    int32_t * healthDamage;
    int32_t * health;
    char ** attacker;
    char ** weapon;
    int32_t * demoTickNumber;
    int32_t * demoFile;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        victimName = (char **) malloc(rows * sizeof(char *));
        armorDamage = (int32_t *) malloc(rows * sizeof(int32_t));
        armor = (int32_t *) malloc(rows * sizeof(int32_t));
        healthDamage = (int32_t *) malloc(rows * sizeof(int32_t));
        health = (int32_t *) malloc(rows * sizeof(int32_t));
        attacker = (char **) malloc(rows * sizeof(char *));
        weapon = (char **) malloc(rows * sizeof(char *));
        demoTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
        demoFile = (int32_t *) malloc(rows * sizeof(int32_t));
    }

    PlayerHurt() { };
    ~PlayerHurt() {
        if (!beenInitialized){
            return;
        }
        free(armorDamage);
        free(armor);
        free(healthDamage);
        free(health);

        for (int64_t row = 0; row < size; row++) {
            free(victimName[row]);
            free(attacker[row]);
            free(weapon[row]);
        }
        free(attacker);
        free(weapon);

        free(demoTickNumber);
        free(demoFile);
    }

    PlayerHurt(const PlayerHurt& other) = delete;
    PlayerHurt& operator=(const PlayerHurt& other) = delete;
};

struct Grenades : public ColStore {
public:
    char ** thrower;
    char ** grenadeType;
    int32_t * demoTickNumber;
    int32_t * demoFile;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        thrower = (char **) malloc(rows * sizeof(char *));
        grenadeType = (char **) malloc(rows * sizeof(char *));
        demoTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
        demoFile = (int32_t *) malloc(rows * sizeof(int32_t));
    }

    Grenades() { };
    ~Grenades() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(thrower[row]);
            free(grenadeType[row]);
        }
        free(thrower);
        free(grenadeType);

        free(demoTickNumber);
        free(demoFile);
    }

    Grenades(const Grenades& other) = delete;
    Grenades& operator=(const Grenades& other) = delete;
};

struct Kills : public ColStore {
public:
    char ** killer;
    char ** victim;
    char ** weapon;
    char ** assister;
    bool * isHeadshot;
    bool * isWallbang;
    int32_t * penetratedObjects;
    int32_t * demoTickNumber;
    int32_t * demoFile;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        killer = (char **) malloc(rows * sizeof(char *));
        victim = (char **) malloc(rows * sizeof(char *));
        weapon = (char **) malloc(rows * sizeof(char *));
        assister = (char **) malloc(rows * sizeof(char *));
        isHeadshot = (bool *) malloc(rows * sizeof(bool));
        isWallbang = (bool *) malloc(rows * sizeof(bool));
        penetratedObjects = (int32_t *) malloc(rows * sizeof(int32_t));
        demoTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
        demoFile = (int32_t *) malloc(rows * sizeof(int32_t));
    }

    Kills() { };
    ~Kills() {
        if (!beenInitialized){
            return;
        }
        free(isHeadshot);
        free(isWallbang);
        free(penetratedObjects);

        for (int64_t row = 0; row < size; row++) {
            free(killer[row]);
            free(victim[row]);
            free(weapon[row]);
            free(assister[row]);
        }
        free(killer);
        free(victim);
        free(weapon);
        free(assister);

        free(demoTickNumber);
        free(demoFile);
    }

    Kills(const Kills& other) = delete;
    Kills& operator=(const Kills& other) = delete;
};

void loadData(Position & position, Spotted & spotted, WeaponFire & weaponFire, PlayerHurt & playerHurt,
               Grenades & grenades, Kills & kills, string dataPath);

#endif //CSKNOW_LOAD_DATA_H
