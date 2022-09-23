#ifndef CSKNOW_LOAD_DATA_H
#define CSKNOW_LOAD_DATA_H
#define NUM_PLAYERS 10
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include "IntervalTree.h"
#include "queries/parser_constants.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
#define ENGINE_TEAM_UNASSIGNED 0
#define ENGINE_TEAM_SPEC 1
#define ENGINE_TEAM_T 2
#define ENGINE_TEAM_CT 3
#define INVALID_ID -1
#define WEAPON_RECOIL_SCALE 2.0
#define VIEW_RECOIL_TRACKING 0.45

struct RangeIndexEntry {
    int64_t minId, maxId;

    string toCSV() const {
        return std::to_string(minId) + "," + std::to_string(maxId);
    }
};

typedef RangeIndexEntry * RangeIndex;
typedef IntervalTree<int64_t, int64_t> IntervalIndex;

class ColStore {
public:
    bool beenInitialized = false;
    int64_t size;
    vector<string> fileNames;
    vector<int64_t> gameStarts;
    vector<int64_t> id;
    virtual void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        beenInitialized = true;
        size = rows;
        fileNames.resize(numFiles);
        this->gameStarts = gameStarts;
        this->id.resize(rows);
    }
};


class Equipment : public ColStore {
public:
    char ** name;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        name = (char **) malloc(rows * sizeof(char*));
    }

    Equipment() { };
    ~Equipment() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(name[row]);
        }
        free(name);
    }

    Equipment(const Equipment& other) = delete;
    Equipment& operator=(const Equipment& other) = delete;
};

class GameTypes : public ColStore {
public:
    char ** tableType;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        tableType = (char **) malloc(rows * sizeof(char*));
    }

    GameTypes() { };
    ~GameTypes() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(tableType[row]);
        }
        free(tableType);
    }

    GameTypes(const GameTypes& other) = delete;
    GameTypes& operator=(const GameTypes& other) = delete;
};

class HitGroups : public ColStore {
public:
    char ** groupName;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        groupName = (char **) malloc(rows * sizeof(char*));
    }

    HitGroups() { };
    ~HitGroups() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(groupName[row]);
        }
        free(groupName);
    }

    HitGroups(const HitGroups& other) = delete;
    HitGroups& operator=(const HitGroups& other) = delete;
};

class Games : public ColStore {
public:
    char ** demoFile;
    double * demoTickRate;
    double * gameTickRate;
    char ** mapName;
    int64_t * gameType;
    RangeIndex roundsPerGame;
    RangeIndex playersPerGame;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        demoFile = (char **) malloc(rows * sizeof(char*));
        demoTickRate = (double *) malloc(rows * sizeof(double));
        gameTickRate = (double *) malloc(rows * sizeof(double));
        mapName = (char **) malloc(rows * sizeof(char*));
        gameType = (int64_t *) malloc(rows * sizeof(int64_t));
        roundsPerGame = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
        playersPerGame = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
    }

    Games() { };
    ~Games() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(demoFile[row]);
        }
        free(demoFile);

        free(demoTickRate);
        free(gameTickRate);
        for (int64_t row = 0; row < size; row++) {
            free(mapName[row]);
        }
        free(mapName);
        free(gameType);
        free(roundsPerGame);
        free(playersPerGame);
    }

    Games(const Games& other) = delete;
    Games& operator=(const Games& other) = delete;
};

class Players : public ColStore {
public:
    int64_t * gameId;
    char ** name;
    int64_t * steamId;
    // add this offset to id to get the row entry
    int64_t idOffset = 1;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        gameId = (int64_t *) malloc(rows * sizeof(int64_t));
        name = (char **) malloc(rows * sizeof(char*));
        steamId = (int64_t *) malloc(rows * sizeof(int64_t));
    }

    Players() { };
    ~Players() {
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

    Players(const Players& other) = delete;
    Players& operator=(const Players& other) = delete;
};

class Rounds : public ColStore {
public:
    int64_t * gameId;
    int64_t * startTick;
    // tick when objective accomplished
    int64_t * endTick;
    // tick when run around time ends
    int64_t * endOfficialTick;
    bool * warmup;
    bool * overtime;
    int64_t * freezeTimeEnd;
    int16_t * roundNumber;
    int16_t * roundEndReason;
    int16_t * winner;
    int16_t * tWins;
    int16_t * ctWins;
    RangeIndex ticksPerRound;
    // add this offset to id to get the row entry
    int64_t idOffset = 1;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        gameId = (int64_t *) malloc(rows * sizeof(int64_t));
        startTick = (int64_t *) malloc(rows * sizeof(int64_t));
        endTick = (int64_t *) malloc(rows * sizeof(int64_t));
        endOfficialTick = (int64_t *) malloc(rows * sizeof(int64_t));
        warmup = (bool *) malloc(rows * sizeof(bool));
        overtime = (bool *) malloc(rows * sizeof(bool));
        freezeTimeEnd = (int64_t *) malloc(rows * sizeof(int64_t));
        roundNumber = (int16_t *) malloc(rows * sizeof(int16_t));
        roundEndReason = (int16_t *) malloc(rows * sizeof(int16_t));
        winner = (int16_t *) malloc(rows * sizeof(int16_t));
        tWins = (int16_t *) malloc(rows * sizeof(int16_t));
        ctWins = (int16_t *) malloc(rows * sizeof(int16_t));
        ticksPerRound = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
    }

    Rounds() { };
    ~Rounds() {
        if (!beenInitialized){
            return;
        }

        free(gameId);
        free(startTick);
        free(endTick);
        free(endOfficialTick);
        free(warmup);
        free(overtime);
        free(freezeTimeEnd);
        free(roundNumber);
        free(roundEndReason);
        free(winner);
        free(tWins);
        free(ctWins);
        free(ticksPerRound);
    }

    Rounds(const Rounds& other) = delete;
    Rounds& operator=(const Rounds& other) = delete;
};

class Ticks: public ColStore {
public:
    int64_t * roundId;
    int64_t * gameTime;
    int64_t * demoTickNumber;
    int64_t * gameTickNumber;
    int64_t * bombCarrier;
    double * bombX;
    double * bombY;
    double * bombZ;
    RangeIndex patPerTick;
    RangeIndex spottedPerTick;
    RangeIndex footstepPerTick;
    IntervalIndex weaponFirePerTick;
    IntervalIndex killsPerTick;
    IntervalIndex hurtPerTick;
    IntervalIndex grenadesPerTick;
    IntervalIndex grenadesThrowPerTick;
    IntervalIndex grenadesActivePerTick;
    IntervalIndex grenadesExpiredPerTick;
    IntervalIndex grenadesDestroyedPerTick;
    IntervalIndex flashedPerTick;
    IntervalIndex plantsPerTick;
    IntervalIndex plantsStartPerTick;
    IntervalIndex plantsEndPerTick;
    IntervalIndex defusalsPerTick;
    IntervalIndex defusalsStartPerTick;
    IntervalIndex defusalsEndPerTick;
    IntervalIndex explosionsPerTick;

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
        patPerTick = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
        spottedPerTick = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
        footstepPerTick = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
    }

    Ticks() { };
    ~Ticks() {
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
        free(patPerTick);
        free(spottedPerTick);
        free(footstepPerTick);
    }

    Ticks(const Ticks& other) = delete;
    Ticks& operator=(const Ticks& other) = delete;

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

class PlayerAtTick: public ColStore {
public:
    int64_t * playerId;
    int64_t * tickId;
    double * posX;
    double * posY;
    double * posZ;
    double * eyePosZ;
    double * velX;
    double * velY;
    double * velZ;
    double * viewX;
    double * viewY;
    double * aimPunchX;
    double * aimPunchY;
    double * viewPunchX;
    double * viewPunchY;
    int16_t * team;
    double * health;
    double * armor;
    bool * hasHelmet;
    bool * isAlive;
    bool * duckingKeyPressed;
    double * duckAmount;
    bool * isWalking;
    bool * isScoped;
    bool * isAirborne;
    double * remainingFlashTime;
    int16_t * activeWeapon;
    int16_t * primaryWeapon;
    int16_t * primaryBulletsClip;
    int16_t * primaryBulletsReserve;
    int16_t * secondaryWeapon;
    int16_t * secondaryBulletsClip;
    int16_t * secondaryBulletsReserve;
    int16_t * numHe;
    int16_t * numFlash;
    int16_t * numSmoke;
    int16_t * numMolotov;
    int16_t * numIncendiary;
    int16_t * numDecoy;
    int16_t * numZeus;
    bool * hasDefuser;
    bool * hasBomb;
    int32_t * money;
    int32_t * ping;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        playerId = (int64_t *) malloc(rows * sizeof(int64_t));
        tickId = (int64_t *) malloc(rows * sizeof(int64_t));
        posX = (double *) malloc(rows * sizeof(double));
        posY = (double *) malloc(rows * sizeof(double));
        posZ = (double *) malloc(rows * sizeof(double));
        eyePosZ = (double *) malloc(rows * sizeof(double));
        velX = (double *) malloc(rows * sizeof(double));
        velY = (double *) malloc(rows * sizeof(double));
        velZ = (double *) malloc(rows * sizeof(double));
        viewX = (double *) malloc(rows * sizeof(double));
        viewY = (double *) malloc(rows * sizeof(double));
        aimPunchX = (double *) malloc(rows * sizeof(double));
        aimPunchY = (double *) malloc(rows * sizeof(double));
        viewPunchX = (double *) malloc(rows * sizeof(double));
        viewPunchY = (double *) malloc(rows * sizeof(double));
        team = (int16_t *) malloc(rows * sizeof(int16_t));
        health = (double *) malloc(rows * sizeof(double));
        armor = (double *) malloc(rows * sizeof(double));
        hasHelmet = (bool *) malloc(rows * sizeof(bool));
        isAlive = (bool *) malloc(rows * sizeof(bool));
        duckingKeyPressed = (bool *) malloc(rows * sizeof(bool));
        duckAmount = (double *) malloc(rows * sizeof(double));
        isWalking = (bool *) malloc(rows * sizeof(bool));
        isScoped = (bool *) malloc(rows * sizeof(bool));
        isAirborne = (bool *) malloc(rows * sizeof(bool));
        remainingFlashTime = (double *) malloc(rows * sizeof(double));
        activeWeapon = (int16_t *) malloc(rows * sizeof(int16_t));
        primaryWeapon = (int16_t *) malloc(rows * sizeof(int16_t));
        primaryBulletsClip = (int16_t *) malloc(rows * sizeof(int16_t));
        primaryBulletsReserve = (int16_t *) malloc(rows * sizeof(int16_t));
        secondaryWeapon = (int16_t *) malloc(rows * sizeof(int16_t));
        secondaryBulletsClip = (int16_t *) malloc(rows * sizeof(int16_t));
        secondaryBulletsReserve = (int16_t *) malloc(rows * sizeof(int16_t));
        numHe = (int16_t *) malloc(rows * sizeof(int16_t));
        numFlash = (int16_t *) malloc(rows * sizeof(int16_t));
        numSmoke = (int16_t *) malloc(rows * sizeof(int16_t));
        numMolotov = (int16_t *) malloc(rows * sizeof(int16_t));
        numIncendiary = (int16_t *) malloc(rows * sizeof(int16_t));
        numDecoy = (int16_t *) malloc(rows * sizeof(int16_t));
        numZeus = (int16_t *) malloc(rows * sizeof(int16_t));
        hasDefuser = (bool *) malloc(rows * sizeof(bool));
        hasBomb = (bool *) malloc(rows * sizeof(bool));
        money = (int32_t *) malloc(rows * sizeof(int32_t));
        ping = (int32_t *) malloc(rows * sizeof(int32_t));
    }

    void makePitchNeg90To90() {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            if (viewY[i] > 260.0) {
                viewY[i] -= 360;
            }
        }
    }


    PlayerAtTick() { };
    ~PlayerAtTick() {
        if (!beenInitialized){
            return;
        }
        free(playerId);
        free(tickId);
        free(posX);
        free(posY);
        free(posZ);
        free(eyePosZ);
        free(velX);
        free(velY);
        free(velZ);
        free(viewX);
        free(viewY);
        free(aimPunchX);
        free(aimPunchY);
        free(viewPunchX);
        free(viewPunchY);
        free(team);
        free(health);
        free(armor);
        free(hasHelmet);
        free(isAlive);
        free(duckingKeyPressed);
        free(duckAmount);
        free(isWalking);
        free(isScoped);
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
        free(ping);
    }

    PlayerAtTick(const PlayerAtTick& other) = delete;
    PlayerAtTick& operator=(const PlayerAtTick& other) = delete;

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

class Spotted : public ColStore {
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

    Spotted() { };
    ~Spotted() {
        if (!beenInitialized){
            return;
        }
        free(tickId);
        free(spottedPlayer);
        free(spotterPlayer);
        free(isSpotted);
    }

    Spotted(const Spotted& other) = delete;
    Spotted& operator=(const Spotted& other) = delete;
};

class Footstep : public ColStore {
public:
    int64_t * tickId;
    int64_t * steppingPlayer;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        tickId = (int64_t *) malloc(rows * sizeof(int64_t));
        steppingPlayer = (int64_t *) malloc(rows * sizeof(int64_t));
    }

    Footstep() { };
    ~Footstep() {
        if (!beenInitialized){
            return;
        }
        free(tickId);
        free(steppingPlayer);
    }

    Footstep(const Footstep& other) = delete;
    Footstep& operator=(const Footstep& other) = delete;
};

class WeaponFire : public ColStore {
public:
    int64_t * tickId;
    int64_t * shooter;
    int16_t * weapon;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        tickId = (int64_t *) malloc(rows * sizeof(int64_t));
        shooter = (int64_t *) malloc(rows * sizeof(int64_t));
        weapon = (int16_t *) malloc(rows * sizeof(int16_t));
    }

    WeaponFire() { };
    ~WeaponFire() {
        if (!beenInitialized){
            return;
        }
        free(tickId);
        free(shooter);
        free(weapon);
    }

    WeaponFire(const WeaponFire& other) = delete;
    WeaponFire& operator=(const WeaponFire& other) = delete;
};

class Kills : public ColStore {
public:
    int64_t * tickId;
    int64_t * killer;
    int64_t * victim;
    int16_t * weapon;
    int64_t * assister;
    bool * isHeadshot;
    bool * isWallbang;
    int32_t * penetratedObjects;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        tickId = (int64_t *) malloc(rows * sizeof(int64_t));
        killer = (int64_t *) malloc(rows * sizeof(int64_t));
        victim = (int64_t *) malloc(rows * sizeof(int64_t));
        weapon = (int16_t *) malloc(rows * sizeof(int16_t));
        assister = (int64_t *) malloc(rows * sizeof(int64_t));
        isHeadshot = (bool *) malloc(rows * sizeof(int64_t));
        isWallbang = (bool *) malloc(rows * sizeof(int64_t));
        penetratedObjects = (int32_t *) malloc(rows * sizeof(int32_t));
    }

    Kills() { };
    ~Kills() {
        if (!beenInitialized){
            return;
        }
        free(tickId);
        free(killer);
        free(victim);
        free(weapon);
        free(assister);
        free(isHeadshot);
        free(isWallbang);
        free(penetratedObjects);
    }

    Kills(const Kills& other) = delete;
    Kills& operator=(const Kills& other) = delete;
};

class Hurt : public ColStore {
public:
    int64_t * tickId;
    int64_t * victim;
    int64_t * attacker;
    DemoEquipmentType * weapon;
    int32_t * armorDamage;
    int32_t * armor;
    int32_t * healthDamage;
    int32_t * health;
    int64_t * hitGroup;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        tickId = (int64_t *) malloc(rows * sizeof(int64_t));
        victim = (int64_t *) malloc(rows * sizeof(int64_t));
        attacker = (int64_t *) malloc(rows * sizeof(int64_t));
        weapon = (DemoEquipmentType *) malloc(rows * sizeof(int16_t));
        armorDamage = (int32_t *) malloc(rows * sizeof(int32_t));
        armor = (int32_t *) malloc(rows * sizeof(int32_t));
        healthDamage = (int32_t *) malloc(rows * sizeof(int32_t));
        health = (int32_t *) malloc(rows * sizeof(int32_t));
        hitGroup = (int64_t *) malloc(rows * sizeof(int64_t));
    }

    Hurt() { };
    ~Hurt() {
        if (!beenInitialized){
            return;
        }
        free(tickId);
        free(victim);
        free(attacker);
        free(weapon);
        free(armorDamage);
        free(armor);
        free(healthDamage);
        free(health);
        free(hitGroup);
    }

    Hurt(const Hurt& other) = delete;
    Hurt& operator=(const Hurt& other) = delete;
};

class Grenades : public ColStore {
public:
    int64_t * thrower;
    int16_t * grenadeType;
    int64_t * throwTick;
    int64_t * activeTick;
    int64_t * expiredTick;
    int64_t * destroyTick;
    RangeIndex flashedPerGrenade;
    RangeIndex trajectoryPerGrenade;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        thrower = (int64_t *) malloc(rows * sizeof(int64_t));
        grenadeType = (int16_t *) malloc(rows * sizeof(int16_t));
        throwTick = (int64_t *) malloc(rows * sizeof(int64_t));
        activeTick = (int64_t *) malloc(rows * sizeof(int64_t));
        expiredTick = (int64_t *) malloc(rows * sizeof(int64_t));
        destroyTick = (int64_t *) malloc(rows * sizeof(int64_t));
        flashedPerGrenade = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
        trajectoryPerGrenade = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
    }

    Grenades() { };
    ~Grenades() {
        if (!beenInitialized){
            return;
        }
        free(thrower);
        free(grenadeType);
        free(throwTick);
        free(activeTick);
        free(expiredTick);
        free(destroyTick);
        free(flashedPerGrenade);
        free(trajectoryPerGrenade);
    }

    Grenades(const Grenades& other) = delete;
    Grenades& operator=(const Grenades& other) = delete;
};

class Flashed : public ColStore {
public:
    int64_t * tickId;
    int64_t * grenadeId;
    int64_t * thrower;
    int64_t * victim;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        tickId = (int64_t *) malloc(rows * sizeof(int64_t));
        grenadeId = (int64_t *) malloc(rows * sizeof(int64_t));
        thrower = (int64_t *) malloc(rows * sizeof(int64_t));
        victim = (int64_t *) malloc(rows * sizeof(int64_t));
    }

    Flashed() { };
    ~Flashed() {
        if (!beenInitialized){
            return;
        }
        free(tickId);
        free(grenadeId);
        free(thrower);
        free(victim);
    }

    Flashed(const Flashed& other) = delete;
    Flashed& operator=(const Flashed& other) = delete;
};

class GrenadeTrajectories : public ColStore {
public:
    int64_t * grenadeId;
    int32_t * idPerGrenade;
    double * posX;
    double * posY;
    double * posZ;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        grenadeId = (int64_t *) malloc(rows * sizeof(int64_t));
        idPerGrenade = (int32_t *) malloc(rows * sizeof(int32_t));
        posX = (double *) malloc(rows * sizeof(double));
        posY = (double *) malloc(rows * sizeof(double));
        posZ = (double *) malloc(rows * sizeof(double));
    }

    GrenadeTrajectories() { };
    ~GrenadeTrajectories() {
        if (!beenInitialized){
            return;
        }
        free(grenadeId);
        free(idPerGrenade);
        free(posX);
        free(posY);
        free(posZ);
    }

    GrenadeTrajectories(const GrenadeTrajectories& other) = delete;
    GrenadeTrajectories& operator=(const GrenadeTrajectories& other) = delete;
};

class Plants : public ColStore {
public:
    int64_t * startTick;
    int64_t * endTick;
    int64_t * planter;
    bool * succesful;
    RangeIndex defusalsPerGrenade;
    RangeIndex explosionsPerGrenade;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        startTick = (int64_t *) malloc(rows * sizeof(int64_t));
        endTick = (int64_t *) malloc(rows * sizeof(int64_t));
        planter = (int64_t *) malloc(rows * sizeof(int64_t));
        succesful = (bool *) malloc(rows * sizeof(bool));
        defusalsPerGrenade = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
        explosionsPerGrenade = (RangeIndexEntry *) malloc(rows * sizeof(RangeIndexEntry));
    }

    Plants() { };
    ~Plants() {
        if (!beenInitialized){
            return;
        }
        free(startTick);
        free(endTick);
        free(planter);
        free(succesful);
        free(defusalsPerGrenade);
        free(explosionsPerGrenade);
    }

    Plants(const Plants& other) = delete;
    Plants& operator=(const Plants& other) = delete;
};

class Defusals : public ColStore {
public:
    int64_t * plantId;
    int64_t * startTick;
    int64_t * endTick;
    int64_t * defuser;
    bool * succesful;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        plantId = (int64_t *) malloc(rows * sizeof(int64_t));
        startTick = (int64_t *) malloc(rows * sizeof(int64_t));
        endTick = (int64_t *) malloc(rows * sizeof(int64_t));
        defuser = (int64_t *) malloc(rows * sizeof(int64_t));
        succesful = (bool *) malloc(rows * sizeof(bool));
    }

    Defusals() { };
    ~Defusals() {
        if (!beenInitialized){
            return;
        }
        free(plantId);
        free(startTick);
        free(endTick);
        free(defuser);
        free(succesful);
    }

    Defusals(const Defusals& other) = delete;
    Defusals& operator=(const Defusals& other) = delete;
};

class Explosions : public ColStore {
public:
    int64_t * plantId;
    int64_t * tickId;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        plantId = (int64_t *) malloc(rows * sizeof(int64_t));
        tickId = (int64_t *) malloc(rows * sizeof(int64_t));
    }

    Explosions() { };
    ~Explosions() {
        if (!beenInitialized){
            return;
        }
        free(plantId);
        free(tickId);
    }

    Explosions(const Explosions& other) = delete;
    Explosions& operator=(const Explosions& other) = delete;
};

void loadData(Equipment & equipment, GameTypes & gameTypes, HitGroups & hitGroups, Games & games, Players & players,
              Rounds & unfilteredRounds, Rounds & filteredRounds, Ticks & ticks, PlayerAtTick & playerAtTick, Spotted & spotted, Footstep & footstep, WeaponFire & weaponFire,
              Kills & kills, Hurt & hurt, Grenades & grenades, Flashed & flashed, GrenadeTrajectories & grenadeTrajectories,
              Plants & plants, Defusals & defusals, Explosions & explosions, string dataPath);

#endif //CSKNOW_LOAD_DATA_H
