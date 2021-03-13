#ifndef CSKNOW_LOAD_DATA_H
#define CSKNOW_LOAD_DATA_H
#define NUM_PLAYERS 10
#include <string>
#include <set>
#include <vector>
using std::string;
using std::vector;
using std::set;

struct OpenFiles {
    set<string> paths;
};

struct PlayerPositionBuilder {
    vector<string> name;
    vector<int8_t> team;
    vector<double> xPosition;
    vector<double> yPosition;
    vector<double> zPosition;
    vector<double> xViewDirection;
    vector<double> yViewDirection;
    vector<bool> isAlive;
    vector<bool> isBlinded;
};

struct PositionBuilder {
    vector<int32_t> demoTickNumber;
    vector<int32_t> gameTickNumber;
    vector<string> demoFile;
    vector<bool> matchStarted;
    vector<int8_t> gamePhase;
    vector<int8_t> roundsPlayed;
    vector<bool> isWarmup;
    vector<bool> roundStart;
    vector<bool> roundEnd;
    vector<int8_t> roundEndReason;
    vector<bool> freezeTimeEnded;
    vector<int8_t> tScore;
    vector<int8_t> ctScore;
    vector<int8_t> numPlayers;
    PlayerPositionBuilder players[NUM_PLAYERS];
};

struct PlayerPosition {
    string * name;
    int8_t * team;
    double * xPosition;
    double * yPosition;
    double * zPosition;
    double * xViewDirection;
    double * yViewDirection;
    bool * isAlive;
    bool * isBlinded;
};

struct Position {
    int64_t size;
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
    string * demoFile;
};

struct Spotted {
    vector<int32_t> demoTickNumber;
    vector<string> demoFile;
    vector<string> spottedPlayer;
    vector<string> player0Name;
    vector<bool> player0Spotter;
    vector<string> player1Name;
    vector<bool> player1Spotter;
    vector<string> player2Name;
    vector<bool> player2Spotter;
    vector<string> player3Name;
    vector<bool> player3Spotter;
    vector<string> player4Name;
    vector<bool> player4Spotter;
    vector<string> player5Name;
    vector<bool> player5Spotter;
    vector<string> player6Name;
    vector<bool> player6Spotter;
    vector<string> player7Name;
    vector<bool> player7Spotter;
    vector<string> player8Name;
    vector<bool> player8Spotter;
    vector<string> player9Name;
    vector<bool> player9Spotter;
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
               Grenades & grenades, Kills & kills, string dataPath, OpenFiles & openFiles);

#endif //CSKNOW_LOAD_DATA_H
