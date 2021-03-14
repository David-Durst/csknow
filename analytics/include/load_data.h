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
    int64_t size;
    vector<string> fileNames;
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
    
    void init(int64_t rows, int64_t numFiles) {
        size = rows;
        fileNames.resize(numFiles);
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
    
    ~Position() {
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
};

class Spotted : public ColStore {
public:
    int32_t * demoTickNumber;
    int32_t * demoFile;
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
    
    void init(int64_t rows, int64_t numFiles) {
        size = rows;
        fileNames.resize(numFiles);
        demoTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
        demoFile = (int32_t *) malloc(rows * sizeof(int32_t));
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
    }
    
    ~Spotted() {
        free(demoTickNumber);
        free(demoFile);
        for (int i = 0; i < NUM_PLAYERS; i++) {
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
        }
        
    }
};

struct WeaponFire {
    vector<int32_t> demoTickNumber;
    vector<string> demoFile;
    vector<string> shooter;
    vector<string> weapon;
};

struct PlayerHurt {
    vector<int32_t> demoTickNumber;
    vector<string> demoFile;
    vector<string> victimName;
    vector<int32_t> armorDamage;
    vector<int32_t> armor;
    vector<int32_t> healthDamage;
    vector<int32_t> health;
    vector<string> attacker;
    vector<string> weapon;
};

struct Grenades {
    vector<int32_t> demoTickNumber;
    vector<string> demoFile;
    vector<string> thrower;
    vector<string> grenadeType;
};

struct Kills {
    vector<int32_t> demoTickNumber;
    vector<string> demoFile;
    vector<string> killer;
    vector<string> victim;
    vector<string> weapon;
    vector<string> assister;
    vector<bool> isHeadshot;
    vector<bool> isWallbang;
    vector<int32_t> penetratedObjects;
};

void loadData(Position & position, Spotted & spotted, WeaponFire & weaponFire, PlayerHurt & playerHurt,
               Grenades & grenades, Kills & kills, string dataPath);

#endif //CSKNOW_LOAD_DATA_H
