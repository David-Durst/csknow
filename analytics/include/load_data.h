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
    vector<string> fileNames;
    vector<int64_t> gameStarts;
    set<int64_t> skipRows;
    virtual void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        beenInitialized = true;
        size = rows;
        fileNames.resize(numFiles);
        this->gameStarts = gameStarts;
    }
};

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

class Position: public ColStore {
public:
    vector<int64_t> firstRowAfterWarmup;
    int32_t * demoTickNumber;
    int32_t * gameTickNumber;
    bool * matchStarted;
    int8_t * gamePhase;
    int8_t * roundsPlayed;
    bool * isWarmup;
    bool * roundStart;
    bool * roundEnd;
    int8_t * roundEndReason;
    bool * freezeTimeEnded;
    int8_t * tScore;
    int8_t * ctScore;
    int8_t * numPlayers;
    PlayerPosition players[NUM_PLAYERS];
    int32_t * demoFile;
    
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        firstRowAfterWarmup.resize(numFiles);
        demoTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
        gameTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
        matchStarted = (bool *) malloc(rows * sizeof(bool));
        gamePhase = (int8_t *) malloc(rows * sizeof(int8_t));
        roundsPlayed = (int8_t *) malloc(rows * sizeof(int8_t));
        isWarmup = (bool *) malloc(rows * sizeof(bool));
        roundStart = (bool *) malloc(rows * sizeof(bool));
        roundEnd = (bool *) malloc(rows * sizeof(bool));
        roundEndReason = (int8_t *) malloc(rows * sizeof(int8_t));
        freezeTimeEnded = (bool *) malloc(rows * sizeof(bool));
        tScore = (int8_t *) malloc(rows * sizeof(int8_t));
        ctScore = (int8_t *) malloc(rows * sizeof(int8_t));
        numPlayers = (int8_t *) malloc(rows * sizeof(int8_t));
        for (int i = 0; i < NUM_PLAYERS; i++) {
            players[i].name = (char **) malloc(rows * sizeof(char*));
            players[i].team = (int8_t *) malloc(rows * sizeof(int8_t));
            players[i].xPosition = (double *) malloc(rows * sizeof(double));
            players[i].yPosition = (double *) malloc(rows * sizeof(double));
            players[i].zPosition = (double *) malloc(rows * sizeof(double));
            players[i].xViewDirection = (double *) malloc(rows * sizeof(double));
            players[i].yViewDirection = (double *) malloc(rows * sizeof(double));
            players[i].isAlive = (bool *) malloc(rows * sizeof(bool));
            players[i].isBlinded = (bool *) malloc(rows * sizeof(bool));
        }
        demoFile = (int32_t *) malloc(rows * sizeof(int32_t*));
    }

    void makePitchNeg90To90() {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            for (int p = 0; p < 10; p++) {
                if (players[p].yViewDirection[i] > 260.0) {
                    players[p].yViewDirection[i] -= 360;
                }
            }
        }
    }


    Position() { };
    ~Position() {
        if (!beenInitialized){
            return;
        }
        free(demoTickNumber);
        free(gameTickNumber);
        free(matchStarted);
        free(gamePhase);
        free(roundsPlayed);
        free(isWarmup);
        free(roundStart);
        free(roundEnd);
        free(roundEndReason);
        free(freezeTimeEnded);
        free(tScore);
        free(ctScore);
        free(numPlayers);

        for (int i = 0; i < NUM_PLAYERS; i++) {
            for (int64_t row = 0; row < size; row++) {
                free(players[i].name[row]);
            }
            free(players[i].name);
            free(players[i].team);
            free(players[i].xPosition);
            free(players[i].yPosition);
            free(players[i].zPosition);
            free(players[i].xViewDirection);
            free(players[i].yViewDirection);
            free(players[i].isAlive);
            free(players[i].isBlinded);
        }
        free(demoFile);
    }

    Position(const Position& other) = delete;
    Position& operator=(const Position& other) = delete;

    // since spotted tracks names for spotted player, need to map that to the player index

    map<string, int> getPlayerNameToIndex(int64_t gameIndex) const {
        map<string, int> result;
        for (int i = 0; i < NUM_PLAYERS; i++) {
            result.insert({players[i].name[firstRowAfterWarmup[gameIndex]], i});
        }
        return result;
    }
};

struct SpotterPlayer {
    char ** playerName;
    bool * playerSpotter;
};

class Spotted : public ColStore {
public:
    char ** spottedPlayer;
    SpotterPlayer spotters[NUM_PLAYERS];
    int32_t * demoTickNumber;
    int32_t * demoFile;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        spottedPlayer = (char **) malloc(rows * sizeof(char*));
        for (int i = 0; i < NUM_PLAYERS; i++) {
            spotters[i].playerName = (char **) malloc(rows * sizeof(char*));
            spotters[i].playerSpotter = (bool *) malloc(rows * sizeof(bool));

        }
        demoTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
        demoFile = (int32_t *) malloc(rows * sizeof(int32_t));
    }

    Spotted() { };
    ~Spotted() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(spottedPlayer[row]);
        }

        for (int i = 0; i < NUM_PLAYERS; i++) {
            for (int64_t row = 0; row < size; row++) {
                free(spotters[i].playerName[row]);
            }
            free(spotters[i].playerName);
            free(spotters[i].playerSpotter);
        }

        free(demoTickNumber);
        free(demoFile);
    }

    Spotted(const Spotted& other) = delete;
    Spotted& operator=(const Spotted& other) = delete;
};

class WeaponFire : public ColStore {
public:
    char ** shooter;
    char ** weapon;
    int32_t * demoTickNumber;
    int32_t * demoFile;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        shooter = (char **) malloc(rows * sizeof(char*));
        weapon = (char **) malloc(rows * sizeof(char*));
        demoTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
        demoFile = (int32_t *) malloc(rows * sizeof(int32_t));
    }

    WeaponFire() { };
    ~WeaponFire() {
        if (!beenInitialized){
            return;
        }
        for (int64_t row = 0; row < size; row++) {
            free(shooter[row]);
            free(weapon[row]);
        }
        free(shooter);
        free(weapon);

        free(demoTickNumber);
        free(demoFile);
    }

    WeaponFire(const WeaponFire& other) = delete;
    WeaponFire& operator=(const WeaponFire& other) = delete;
};

class PlayerHurt : public ColStore {
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
