//
// Created by durst on 1/25/22.
//

#ifndef CSKNOW_LOAD_SAVE_BOT_DATA_H
#define CSKNOW_LOAD_SAVE_BOT_DATA_H
#include "load_data.h"
#include "geometry.h"
#include "bot_input_bits.h"
#include <iostream>
#include <sstream>


class ServerState {
private:
    void loadClients(string clientsFilePath);
    void loadClientStates(string clientStatesFilePath);

public:
    struct Client {
        int serverId;
        string name;
        bool isBot;
        int32_t lastFrame;
        float lastEyePosX;
        float lastEyePosY;
        float lastEyePosZ;
        float lastFootPosZ;
        float lastEyeAngleX;
        float lastEyeAngleY;
        float lastAimpunchAngleX;
        float lastAimpunchAngleY;
        float lastEyeWithRecoilAngleX;
        float lastEyeWithRecoilAngleY;
        bool isAlive;

        // keyboard/mouse inputs sent to game engine
        int32_t buttons;
        // these range from -1 to 1
        float inputAngleDeltaPctX;
        float inputAngleDeltaPctY;
    };

    vector<int> serverClientIdToCSKnowId;
    vector<Client> clients;
    vector<bool> inputsValid;

    // state for caller to debug
    bool loadedSuccessfully;
    string badPath;
    int numInputLines;
    string inputsCopy;

    void loadServerState(string dataPath);
    void saveBotInputs(string dataPath);
};


#endif //CSKNOW_LOAD_SAVE_BOT_DATA_H
