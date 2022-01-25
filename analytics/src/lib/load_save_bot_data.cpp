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

void ServerState::loadClients(string clientsFilePath) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(clientsFilePath);

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
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].serverId);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, clients[arrayEntry].name);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].isBot);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 3;
    }
    closeMMapFile({fd, stats, file});
}

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
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyePosX);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyePosY);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyePosZ);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastFootPosZ);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeAngleX);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeAngleY);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastAimpunchAngleX);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastAimpunchAngleY);
        }
        else if (colNumber == 9) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeWithRecoilAngleX);
        }
        else if (colNumber == 10) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeWithRecoilAngleY);
        }
        else if (colNumber == 11) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].isAlive);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 12;
    }
    closeMMapFile({fd, stats, file});
}

void ServerState::loadServerState(string dataPath) {
    string clientsFileName = "clients.csv";
    string clientsFilePath = dataPath + "/" + clientsFileName;
    string tmpClientsFileName = "clients.csv.tmp.read";
    string tmpClientsFilePath = dataPath + "/" + tmpClientsFileName;
    string clientStatesFileName = "state.csv";
    string clientStatesFilePath = dataPath + "/" + stateFileName;
    string tmpClientStatesFileName = "state.csv.tmp.read";
    string tmpClientStatesFilePath = dataPath + "/" + stateFileName;

    if (std::filesystem::exists(clientsFilePath) && 
            std::filesystem::exists(clientStatesFilePath)) {
        std::filesystem::rename(clientsFilePath, tmpClientsFilePath);
        std::filesystem::rename(clientStatesFilePath, tmpClientStatesFilePath);
    }

    vector<int64_t> startingPointPerFile = getFileStartingRows({clientsFilePath});
    int64_t rows = startingPointPerFile[1];

    clients.resize(rows);

    loadClients(clientsFilePath);
    loadClientStates(clientStatesFilePath);

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
        serverClientIdToCSKnowId[clients.serverId] = i;
    }
}

void ServerState::saveBotInputs(string dataPath) const {
    string inputsFileName = "input.csv";
    string inputsFilePath = dataPath + "/" + inputsFileName;
    string tmpInputsFileName = "input.csv.tmp.write";
    string tmpInputsFilePath = dataPath + "/" + tmpClientsFileName;

    if (std::filesystem::exists(inputsFilePath)) {
        std::filesystem::rename(inputsFilePath, tmpInputsFilePath);
    }

    inputsCopy.str(string());
    inputsCopy << "Player Index,Buttons,Input Angle Delta Pct X,Input Angle Delta Pct Y\n";
    numLines = 1;

    for (int i = 0; i < (int) inputsValid.size(); i++) {
        if (i < (int) clients.size() && inputsValid[i]) {
            inputsCopy << clients[i].serverId << ","
                << clients[i].buttons << ","
                << clients[i].inputAngleDeltaPctX << ","
                << clients[i].inputAngleDeltaPctY << "\n";
            numLines++;
        }
    }

    std::ofstream fsInputs(tmpInputsFilePath);
    fsInputs << inputsCopy.str();
    fsInputs.close();
}
