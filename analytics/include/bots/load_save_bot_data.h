//
// Created by durst on 1/25/22.
//

#ifndef CSKNOW_LOAD_SAVE_BOT_DATA_H
#define CSKNOW_LOAD_SAVE_BOT_DATA_H
#define MAX_ONE_DIRECTION_ANGLE_VEL 15.0
#define MAX_ONE_DIRECTION_ANGLE_ACCEL 5.0
#define MAX_PITCH_MAGNITUDE 89.0
#include "load_data.h"
#include "geometry.h"
#include <iostream>
#include <sstream>
#include <set>
#include <utility>
#include <chrono>

typedef int64_t CSKnowId;
typedef int64_t CSGOId;
typedef int32_t TeamId;
typedef int32_t RoundNumber;
typedef std::chrono::time_point<std::chrono::system_clock> CSKnowTime;
typedef std::chrono::time_point<std::filesystem::__file_clock> CSGOFileTime;
constexpr CSKnowTime defaultTime = std::chrono::system_clock::time_point();
static inline __attribute__((always_inline))
CSKnowTime secondsToCSKnowTime(double seconds, CSKnowTime initTime = defaultTime) {
    return std::chrono::duration_cast<std::chrono::system_clock::duration>(
        std::chrono::duration<double, std::ratio<1>>(seconds)) + initTime;
}
// long poll up until this time before the next write
constexpr double longPollBufferSeconds = 0.00150;
// poll length
constexpr double longPollSeconds = 0.001;
constexpr double shortPollSeconds = 0.00025;
constexpr double defaultTickInterval = 0.1;

class ServerState {
private:
    void loadGeneralState(const string& generalFilePath);
    void loadClientStates(const string& clientStatesFilePath);
    void loadVisibilityClientPairs(const string& visibilityFilePath);
    void loadC4State(const string& c4FilePath);
    void loadHurtEvents(const string& hurtFilePath);
    void loadWeaponFireEvents(const string& weaponFireFilePath);
    void loadRoundStartEvents(const string& roundStartFilePath);

public:
    string mapName;
    RoundNumber roundNumber;
    int32_t tScore, ctScore;
    int32_t mapNumber;
    double tickInterval = defaultTickInterval, gameTime;
    CSKnowTime loadTime;
    //const static CSKnowTime defaultTime = std::chrono::system_clock::from_time_t(0);

    struct Client {
        int32_t lastFrame;
        int32_t csgoId;
        int32_t lastTeleportId;
        string name;
        TeamId team;
        int health;
        int armor;
        bool hasHelmet;
        int32_t currentWeaponId;
        float nextPrimaryAttack;
        float nextSecondaryAttack;
        float timeWeaponIdle;
        float recoilIndex;
        bool reloadVisuallyComplete;
        int32_t rifleId;
        int32_t rifleClipAmmo;
        int32_t rifleReserveAmmo;
        int32_t pistolId;
        int32_t pistolClipAmmo;
        int32_t pistolReserveAmmo;
        int32_t flashes;
        int32_t molotovs;
        int32_t smokes;
        int32_t hes;
        int32_t decoys;
        int32_t incendiaries;
        int32_t zeus;
        bool hasC4;
        float lastEyePosX;
        float lastEyePosY;
        float lastEyePosZ;
        float lastFootPosZ;
        float lastVelX;
        float lastVelY;
        float lastVelZ;
        // I USE X FOR YAW, Y FOR PITCH, ENGINE DOES OPPOSITE
        float lastEyeAngleX;
        float lastEyeAngleY;
        float lastAimpunchAngleX;
        float lastAimpunchAngleY;
        float lastViewpunchAngleX;
        float lastViewpunchAngleY;
        float lastEyeWithRecoilAngleX;
        float lastEyeWithRecoilAngleY;
        bool isAlive;
        bool isBot;
        bool isAirborne;
        bool isScoped;
        float duckAmount;
        bool duckKeyPressed;
        bool isReloading;
        bool isWalking;
        float flashDuration;
        bool hasDefuser;
        int money;
        int ping;
        float gameTime;
        bool inputSet;

        // keyboard/mouse inputs sent to game engine
        int32_t buttons;
        bool intendedToFire;

        // default initialize this one since it isn't read from file
        int32_t lastTeleportConfirmationId = 0;
        // this tells the aim model whether it can use this value, because reading from game state
        // leads to 1 frame delay oddities due to 1 frame delay
        bool inputAngleDefined = false;
        float inputAngleX;
        float inputAngleY;
        bool inputAngleAbsolute;
        bool forceInput;

        [[nodiscard]]
        Vec2 getCurrentViewAngles() const {
            return {lastEyeAngleX, lastEyeAngleY};
        }

        [[nodiscard]]
        Vec2 getCurrentViewAnglesWithAimpunch() const {
            return {
                lastEyeAngleX + lastAimpunchAngleX,
                lastEyeAngleY + lastAimpunchAngleY
            };
        }

        [[nodiscard]]
        Vec3 getFootPosForPlayer() const {
            return {lastEyePosX, lastEyePosY, lastFootPosZ};
        }

        [[nodiscard]]
        Vec3 getEyePosForPlayer() const {
            return {lastEyePosX, lastEyePosY, lastEyePosZ};
        }

        [[nodiscard]]
        Vec3 getVelocity() const {
            return {lastVelX, lastVelY, lastVelZ};
        }
    };

    set<CSGOId> csgoIds;
    vector<int> csgoIdToCSKnowId;
    vector<Client> clients;
    [[nodiscard]]
    const ServerState::Client & getClientSlowSafe(CSGOId csgoId) const;
    [[nodiscard]]
    const ServerState::Client & getClient(CSGOId csgoId) const {
        int csknowId = csgoIdToCSKnowId[csgoId];
        return clients[csknowId];
    }
    [[nodiscard]]
    ServerState::Client & getClient(CSGOId csgoId) {
        int csknowId = csgoIdToCSKnowId[csgoId];
        return clients[csknowId];
    }
    [[nodiscard]]
    string getPlayerString(CSGOId playerId) const {
        if (playerId >= 0 && playerId < static_cast<int64_t>(csgoIdToCSKnowId.size())) {
            return "(" + std::to_string(playerId) + ") " + getClient(playerId).name;
        }
        else {
            return "(" + std::to_string(playerId) + ") INVALID";
        }
    }

    vector<bool> inputsValid;
    void setInputs(CSGOId csgoId, int32_t lastTeleportConfirmationId, int32_t buttons, bool intendedToFire,
                   float inputAngleX, float inputAngleY, bool inputAngleAbsolute, bool forceInput) {
        int csknowId = csgoIdToCSKnowId[csgoId];
        Client & curClient = clients[csknowId];
        curClient.lastTeleportConfirmationId = lastTeleportConfirmationId;
        curClient.buttons = buttons;
        curClient.intendedToFire = intendedToFire;
        curClient.inputAngleX = inputAngleX;
        curClient.inputAngleY = inputAngleY;
        curClient.inputAngleAbsolute = inputAngleAbsolute;
        curClient.inputAngleDefined = true;
        curClient.forceInput = forceInput;
        inputsValid[csknowId] = true;
    }

    // visibility state
    std::set<std::pair<int32_t, int32_t>> visibilityClientPairs;
    [[nodiscard]]
    bool isVisible(CSGOId src, CSGOId target) const {
        return visibilityClientPairs.find({std::min(src, target), std::max(src, target)}) != visibilityClientPairs.end();
    }
    [[nodiscard]]
    vector<std::reference_wrapper<const ServerState::Client>> getVisibleEnemies(CSGOId srcId) const {
        const ServerState::Client & srcClient = getClient(srcId);
        vector<std::reference_wrapper<const ServerState::Client>> visibleEnemies;
        if (srcClient.isAlive) {
            for (const auto & otherClient : clients) {
                if (otherClient.team != srcClient.team && otherClient.isAlive &&
                    isVisible(srcClient.csgoId, otherClient.csgoId)) {
                    visibleEnemies.push_back(otherClient);
                }
            }
        }
        return visibleEnemies;
    }

    [[nodiscard]]
    int numPlayersAlive() const {
        int result = 0;
        for (const auto & client : clients) {
            if (client.isAlive) {
                result++;
            }
        }
        return result;
    }

    [[nodiscard]]
    vector<CSGOId> getPlayersOnTeam(int32_t team) const {
        vector<CSGOId> result;
        for (const auto & client : clients) {
            if (client.team == team) {
                result.push_back(client.csgoId);
            }
        }
        return result;
    }

    [[nodiscard]] static
    double getSecondsBetweenTimes(CSKnowTime startTime, CSKnowTime endTime) {
        std::chrono::duration<double> diffTime = endTime - startTime;
        return diffTime.count();
    }

    [[nodiscard]]
    int32_t getLastFrame() const {
        if (clients.empty()) {
            return INVALID_ID;
        }
        else {
            return clients.front().lastFrame;
        }
    }

    // c4 state
    // Is Planted,Pos X,Pos Y,Pos Z
    bool c4Exists;
    bool c4IsPlanted;
    bool c4IsDropped;
    bool c4IsDefused;
    float c4X, c4Y, c4Z;
    int64_t ticksSinceLastPlant;


    // event state
    struct Hurt {
        int64_t victimId;
        int64_t attackerId;
        int32_t health;
        int32_t armor;
        int32_t healthDamage;
        int32_t armorDamage;
        int64_t hitGroup;
        string weapon;

        string toString() const {
            std::ostringstream ss;
            ss << "victim id: " << victimId
                << ", attacker id: " << attackerId
                << ", health: " << health
                << ", armor: " << armor
                << ", health damage: " << healthDamage
                << ", armor damage: " << armorDamage
                << ", hit group: " << hitGroup
                << ", weapon: " << weapon;
            return ss.str();
        }
    };
    vector<Hurt> hurtEvents;

    // event state
    struct WeaponFire {
        int64_t shooter;
        string weapon;
    };
    vector<WeaponFire> weaponFireEvents;
    int64_t lastRoundStartFrame = -1;
    bool newRoundStart = false;

    // state for caller to debug
    bool loadedSuccessfully;
    string badPath;
    int numInputLines, numThinkLines;
    string inputsLog, thinkLog;

    string dataPath;
    void sleepUntilServerStateExists(CSGOFileTime lastFileTime);
    double getGeneralStatFileTime();
    CSGOFileTime loadServerState();
    void setClientIdTrackers();
    void saveBotInputs();
    bool saveScript(const vector<string>& scriptLines) const;
    [[nodiscard]] Vec3 getC4Pos() const;
};

#endif //CSKNOW_LOAD_SAVE_BOT_DATA_H
