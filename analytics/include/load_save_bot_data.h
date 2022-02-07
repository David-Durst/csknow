//
// Created by durst on 1/25/22.
//

#ifndef CSKNOW_LOAD_SAVE_BOT_DATA_H
#define CSKNOW_LOAD_SAVE_BOT_DATA_H
#define MAX_ONE_DIRECTION_ANGLE_VEL 15.0
#define MAX_ONE_DIRECTION_ANGLE_ACCEL 5.0
#define MAX_PITCH_MAGNITUDE 89.0
#define INVALID_SERVER_ID -1
#include "load_data.h"
#include "geometry.h"
#include <iostream>
#include <sstream>
#include <set>
#include <utility>


class ServerState {
private:
    void loadGeneralState(string generalFilePath);
    void loadClientStates(string clientStatesFilePath);
    void loadVisibilityClientPairs(string visibilityFilePath);
    void loadC4State(string c4FilePath);

public:
    string mapName;
    int32_t roundNumber;

    struct Client {
        int32_t lastFrame;
        int32_t serverId;
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
    };

    vector<int> serverClientIdToCSKnowId;
    vector<Client> clients;
    vector<bool> inputsValid;

    // visibility state
    std::set<std::pair<int32_t, int32_t>> visibilityClientPairs;

    // c4 state
    // Is Planted,Pos X,Pos Y,Pos Z
    bool c4IsPlanted;
    bool c4IsDropped;
    float c4X, c4Y, c4Z;


    // state for caller to debug
    bool loadedSuccessfully;
    string badPath;
    int numInputLines, numThinkLines;
    string inputsCopy, thinkCopy;

    void loadServerState(string dataPath);
    void saveBotInputs(string dataPath);
};


#endif //CSKNOW_LOAD_SAVE_BOT_DATA_H
