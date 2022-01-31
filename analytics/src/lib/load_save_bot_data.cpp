//
// Created by durst on 12/29/21.
//
#include "load_save_bot_data.h"
#include "load_cover.h"
#include "load_data.h"
#include "file_helpers.h"
#include <string>
#include <filesystem>
#include <fstream>

void ServerState::loadClientStates(string clientStatesFilePath) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(clientStatesFilePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = 0;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastFrame);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].serverId);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, clients[arrayEntry].name);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].team);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].currentWeaponId);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].rifleId);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].rifleClipAmmo);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].rifleReserveAmmo);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].pistolId);
        }
        else if (colNumber == 9) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].pistolClipAmmo);
        }
        else if (colNumber == 10) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].pistolReserveAmmo);
        }
        else if (colNumber == 11) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].flashes);
        }
        else if (colNumber == 12) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].molotovs);
        }
        else if (colNumber == 13) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].smokes);
        }
        else if (colNumber == 14) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].hes);
        }
        else if (colNumber == 15) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].decoys);
        }
        else if (colNumber == 16) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].incendiaries);
        }
        else if (colNumber == 17) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyePosX);
        }
        else if (colNumber == 18) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyePosY);
        }
        else if (colNumber == 19) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyePosZ);
        }
        else if (colNumber == 20) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastFootPosZ);
        }
        // Y AND X ARE INTENTIONALLY FLIPPED, I USE X FOR YAW, Y FOR PITCH, ENGINE DOES OPPOSITE
        else if (colNumber == 21) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeAngleY);
        }
        else if (colNumber == 22) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeAngleX);
        }
        else if (colNumber == 23) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastAimpunchAngleY);
        }
        else if (colNumber == 24) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastAimpunchAngleX);
        }
        else if (colNumber == 25) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeWithRecoilAngleY);
        }
        else if (colNumber == 26) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeWithRecoilAngleX);
        }
        else if (colNumber == 27) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].isAlive);
        }
        else if (colNumber == 28) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].isBot);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 29;
    }
    closeMMapFile({fd, stats, file});
}

void ServerState::loadServerState(string dataPath) {
    string clientStatesFileName = "state.csv";
    string clientStatesFilePath = dataPath + "/" + clientStatesFileName;
    string tmpClientStatesFileName = "state.csv.tmp.read";
    string tmpClientStatesFilePath = dataPath + "/" + tmpClientStatesFileName;

    if (std::filesystem::exists(clientStatesFilePath)) {
        std::filesystem::rename(clientStatesFilePath, tmpClientStatesFilePath);
        loadedSuccessfully = true;
    }
    else {
        badPath = clientStatesFilePath;
        loadedSuccessfully = false;
        return;
    }

    vector<int64_t> startingPointPerFile = getFileStartingRows({tmpClientStatesFilePath});
    int64_t rows = startingPointPerFile[1];

    clients.resize(rows);
    inputsValid.resize(rows, false);

    loadClientStates(tmpClientStatesFilePath);

    // build map from server id to CSKnow id
    int maxServerId = -1;
    for (const auto & client : clients) {
        maxServerId = std::max(maxServerId, client.serverId);
    }
    serverClientIdToCSKnowId.resize(maxServerId + 1);
    for (int i = 0; i <= maxServerId; i++) {
        serverClientIdToCSKnowId[i] = -1;
    }
    for (int i = 0; i < (int) clients.size(); i++) {
        serverClientIdToCSKnowId[clients[i].serverId] = i;
    }
}

void ServerState::saveBotInputs(string dataPath) {
    string inputsFileName = "input.csv";
    string inputsFilePath = dataPath + "/" + inputsFileName;
    string tmpInputsFileName = "input.csv.tmp.write";
    string tmpInputsFilePath = dataPath + "/" + tmpInputsFileName;

    if (std::filesystem::exists(inputsFilePath)) {
        std::filesystem::rename(inputsFilePath, tmpInputsFilePath);
    }

    std::stringstream inputsStream;
    inputsStream << "Player Index,Buttons,Input Angle Delta Pct Pitch,Input Angle Delta Pct Yaw\n";
    // two lines as need to get rid of the last newline
    numInputLines = 2;

    for (int i = 0; i < (int) inputsValid.size(); i++) {
        if (i < (int) clients.size() && inputsValid[i]) {
            inputsStream << clients[i].serverId << ","
                << clients[i].buttons << ","
                // FLIPPING TO MATCH YAW AND PITCH
                << clients[i].inputAngleDeltaPctY << ","
                << clients[i].inputAngleDeltaPctX << "\n";
            numInputLines++;
        }
    }

    inputsCopy = inputsStream.str();

    std::ofstream fsInputs(tmpInputsFilePath);
    fsInputs << inputsCopy;
    fsInputs.close();
    
    std::filesystem::rename(tmpInputsFilePath, inputsFilePath);
}
