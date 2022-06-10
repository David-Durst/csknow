//
// Created by durst on 1/25/22.
//

#ifndef CSKNOW_LOAD_SAVE_BOT_DATA_H
#define CSKNOW_LOAD_SAVE_BOT_DATA_H
#define MAX_ONE_DIRECTION_ANGLE_VEL 15.0
#define MAX_ONE_DIRECTION_ANGLE_ACCEL 5.0
#define MAX_PITCH_MAGNITUDE 89.0
#define ENGINE_TEAM_T 2
#define ENGINE_TEAM_CT 3
#include "load_data.h"
#include "geometry.h"
#include <iostream>
#include <sstream>
#include <set>
#include <utility>

typedef int64_t CSKnowId;
typedef int64_t CSGOId;

class ServerState {
private:
    void loadGeneralState(string generalFilePath);
    void loadClientStates(string clientStatesFilePath);
    void loadVisibilityClientPairs(string visibilityFilePath);
    void loadC4State(string c4FilePath);

public:
    string mapName;
    int32_t roundNumber;
    int32_t mapNumber;
    double tickInterval;

    struct Client {
        int32_t lastFrame;
        int32_t csgoId;
        string name;
        int32_t team;
        int32_t currentWeaponId;
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
        bool hasC4;
        float lastEyePosX;
        float lastEyePosY;
        float lastEyePosZ;
        float lastFootPosZ;
        float lastFootPosOtherZ;
        float lastVelX;
        float lastVelY;
        float lastVelZ;
        // I USE X FOR YAW, Y FOR PITCH, ENGINE DOES OPPOSITE
        float lastEyeAngleX;
        float lastEyeAngleY;
        float lastAimpunchAngleX;
        float lastAimpunchAngleY;
        float lastEyeWithRecoilAngleX;
        float lastEyeWithRecoilAngleY;
        bool isAlive;
        bool isBot;

        // keyboard/mouse inputs sent to game engine
        int32_t buttons;

        // these range from -1 to 1
        float inputAngleDeltaPctX;
        float inputAngleDeltaPctY;


        Vec2 getCurrentViewAnglesWithAimpunch() const {
            return {
                lastEyeAngleX + lastAimpunchAngleX,
                lastEyeAngleY + lastAimpunchAngleY
            };
        }

        Vec3 getFootPosForPlayer() const {
            return {lastEyePosX, lastEyePosY, lastFootPosZ};
        }

        Vec3 getEyePosForPlayer() const {
            return {lastEyePosX, lastEyePosY, lastEyePosZ};
        }

        Vec3 getVelocity() const {
            return {lastVelX, lastVelY, lastVelZ};
        }
    };

    vector<int> csgoIdToCSKnowId;
    vector<Client> clients;
    const ServerState::Client & getClient(CSGOId csgoId) const {
        int csknowId = csgoIdToCSKnowId[csgoId];
        return clients[csknowId];
    }
    ServerState::Client & getClient(CSGOId csgoId) {
        int csknowId = csgoIdToCSKnowId[csgoId];
        return clients[csknowId];
    }
    string getPlayerString(CSGOId playerId) const {
        if (playerId >= 0 && playerId < csgoIdToCSKnowId.size()) {
            return "(" + std::to_string(playerId) + ") " + getClient(playerId).name;
        }
        else {
            return "(" + std::to_string(playerId) + ") INVALID";
        }
    }

    vector<bool> inputsValid;
    void setInputs(CSGOId csgoId, int32_t buttons, float inputAngleDeltaPctX, float inputAngleDeltaPctY) {
        int csknowId = csgoIdToCSKnowId[csgoId];
        Client & curClient = clients[csknowId];
        curClient.buttons = buttons;
        curClient.inputAngleDeltaPctX = inputAngleDeltaPctX;
        curClient.inputAngleDeltaPctY = inputAngleDeltaPctY;
        inputsValid[csknowId] = true;
    }
    void resetInputs() {
        for (size_t i = 0; i < inputsValid.size(); i++) {
            inputsValid[i] = false;
        }
    }

    // visibility state
    std::set<std::pair<int32_t, int32_t>> visibilityClientPairs;
    bool isVisible(CSGOId src, CSGOId target) const {
        return visibilityClientPairs.find({std::min(src, target), std::max(src, target)}) != visibilityClientPairs.end();
    }

    // c4 state
    // Is Planted,Pos X,Pos Y,Pos Z
    bool c4Exists;
    bool c4IsPlanted;
    bool c4IsDropped;
    float c4X, c4Y, c4Z;


    // state for caller to debug
    bool loadedSuccessfully;
    string badPath;
    int numInputLines, numThinkLines;
    string inputsLog, thinkLog;

    double roundTimeRemaining, timeSincePlant;

    void loadServerState(string dataPath);
    void saveBotInputs(string dataPath);
    Vec3 getC4Pos() const;
};


#endif //CSKNOW_LOAD_SAVE_BOT_DATA_H
