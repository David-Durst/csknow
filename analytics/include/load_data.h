#ifndef CSKNOW_LOAD_DATA_H
#define CSKNOW_LOAD_DATA_H
#define NUM_PLAYERS 10
#include <string>
#include <set>
#include <vector>
#include <iostream>
using std::string;
using std::vector;
using std::set;

class ColStore {
public:
    bool beenInitialized = false;
    int64_t size;
    vector<string> fileNames;
    vector<int64_t> gameStarts;
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
};

class Spotted : public ColStore {
public:
    char ** spottedPlayer;
    char ** player0Name;
    bool * player0Spotter;
    char ** player1Name;
    bool * player1Spotter;
    char ** player2Name;
    bool * player2Spotter;
    char ** player3Name;
    bool * player3Spotter;
    char ** player4Name;
    bool * player4Spotter;
    char ** player5Name;
    bool * player5Spotter;
    char ** player6Name;
    bool * player6Spotter;
    char ** player7Name;
    bool * player7Spotter;
    char ** player8Name;
    bool * player8Spotter;
    char ** player9Name;
    bool * player9Spotter;
    int32_t * demoTickNumber;
    int32_t * demoFile;

    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        ColStore::init(rows, numFiles, gameStarts);
        spottedPlayer = (char **) malloc(rows * sizeof(char*));
        player0Name = (char **) malloc(rows * sizeof(char*));
        player0Spotter = (bool *) malloc(rows * sizeof(bool));
        player1Name = (char **) malloc(rows * sizeof(char*));
        player1Spotter = (bool *) malloc(rows * sizeof(bool));
        player2Name = (char **) malloc(rows * sizeof(char*));
        player2Spotter = (bool *) malloc(rows * sizeof(bool));
        player3Name = (char **) malloc(rows * sizeof(char*));
        player3Spotter = (bool *) malloc(rows * sizeof(bool));
        player4Name = (char **) malloc(rows * sizeof(char*));
        player4Spotter = (bool *) malloc(rows * sizeof(bool));
        player5Name = (char **) malloc(rows * sizeof(char*));
        player5Spotter = (bool *) malloc(rows * sizeof(bool));
        player6Name = (char **) malloc(rows * sizeof(char*));
        player6Spotter = (bool *) malloc(rows * sizeof(bool));
        player7Name = (char **) malloc(rows * sizeof(char*));
        player7Spotter = (bool *) malloc(rows * sizeof(bool));
        player8Name = (char **) malloc(rows * sizeof(char*));
        player8Spotter = (bool *) malloc(rows * sizeof(bool));
        player9Name = (char **) malloc(rows * sizeof(char*));
        player9Spotter = (bool *) malloc(rows * sizeof(bool));
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
            free(player0Name[row]);
            free(player1Name[row]);
            free(player2Name[row]);
            free(player3Name[row]);
            free(player4Name[row]);
            free(player5Name[row]);
            free(player6Name[row]);
            free(player7Name[row]);
            free(player8Name[row]);
            free(player9Name[row]);
        }
        free(spottedPlayer);
        free(player0Name);
        free(player0Spotter);
        free(player1Name);
        free(player1Spotter);
        free(player2Name);
        free(player2Spotter);
        free(player3Name);
        free(player3Spotter);
        free(player4Name);
        free(player4Spotter);
        free(player5Name);
        free(player5Spotter);
        free(player6Name);
        free(player6Spotter);
        free(player7Name);
        free(player7Spotter);
        free(player8Name);
        free(player8Spotter);
        free(player9Name);
        free(player9Spotter);

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
