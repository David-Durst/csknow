//
// Created by durst on 12/29/21.
//
#include "bots/load_save_bot_data.h"
#include "load_cover.h"
#include "load_data.h"
#include "file_helpers.h"
#include <string>
#include <filesystem>
#include <fstream>

void ServerState::loadGeneralState(const string& generalFilePath) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(generalFilePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < static_cast<size_t>(stats.st_size);
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, mapName);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, roundNumber);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, tScore);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, ctScore);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, mapNumber);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, tickInterval);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, gameTime);
            rowNumber++;
        }
        colNumber = (colNumber + 1) % 7;
    }
    closeMMapFile({fd, stats, file});
}

void ServerState::loadClientStates(const string& clientStatesFilePath) {
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
         curDelimiter < static_cast<size_t>(stats.st_size);
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastFrame);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].csgoId);
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
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].nextPrimaryAttack);
        }
        else if (colNumber == 6) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].nextSecondaryAttack);
        }
        else if (colNumber == 7) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].timeWeaponIdle);
        }
        else if (colNumber == 8) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].reloadVisuallyComplete);
        }
        else if (colNumber == 9) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].rifleId);
        }
        else if (colNumber == 10) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].rifleClipAmmo);
        }
        else if (colNumber == 11) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].rifleReserveAmmo);
        }
        else if (colNumber == 12) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].pistolId);
        }
        else if (colNumber == 13) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].pistolClipAmmo);
        }
        else if (colNumber == 14) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].pistolReserveAmmo);
        }
        else if (colNumber == 15) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].flashes);
        }
        else if (colNumber == 16) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].molotovs);
        }
        else if (colNumber == 17) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].smokes);
        }
        else if (colNumber == 18) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].hes);
        }
        else if (colNumber == 19) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].decoys);
        }
        else if (colNumber == 20) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].incendiaries);
        }
        else if (colNumber == 21) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].hasC4);
        }
        else if (colNumber == 22) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyePosX);
        }
        else if (colNumber == 23) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyePosY);
        }
        else if (colNumber == 24) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyePosZ);
        }
        else if (colNumber == 25) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastFootPosZ);
        }
        else if (colNumber == 26) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastVelX);
        }
        else if (colNumber == 27) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastVelY);
        }
        else if (colNumber == 28) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastVelZ);
        }
        // Y AND X ARE INTENTIONALLY FLIPPED, I USE X FOR YAW, Y FOR PITCH, ENGINE DOES OPPOSITE
        else if (colNumber == 29) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeAngleY);
        }
        else if (colNumber == 30) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeAngleX);
        }
        else if (colNumber == 31) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastAimpunchAngleY);
        }
        else if (colNumber == 32) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastAimpunchAngleX);
        }
        else if (colNumber == 33) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastViewpunchAngleY);
        }
        else if (colNumber == 34) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastViewpunchAngleX);
        }
        else if (colNumber == 35) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeWithRecoilAngleY);
        }
        else if (colNumber == 36) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].lastEyeWithRecoilAngleX);
        }
        else if (colNumber == 37) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].isAlive);
        }
        else if (colNumber == 38) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].isBot);
        }
        else if (colNumber == 39) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].isAirborne);
        }
        else if (colNumber == 40) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].isScoped);
        }
        else if (colNumber == 41) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, clients[arrayEntry].duckAmount);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 42;
    }
    closeMMapFile({fd, stats, file});
}

void ServerState::loadVisibilityClientPairs(const string& visibilityFilePath) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(visibilityFilePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    int32_t visClients[2];
    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < static_cast<size_t>(stats.st_size);
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, visClients[0]);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, visClients[1]);
            visibilityClientPairs.insert({visClients[0], visClients[1]});
            rowNumber++;
        }
        colNumber = (colNumber + 1) % 2;
    }
    closeMMapFile({fd, stats, file});
}

void ServerState::loadC4State(const string& visibilityFilePath) {
    // mmap the file
    auto [fd, stats, file] = openMMapFile(visibilityFilePath);

    // skip the header
    size_t firstRow = getNewline(file, 0, stats.st_size);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    c4Exists = false;

    for (size_t curStart = firstRow + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < static_cast<size_t>(stats.st_size);
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            c4Exists = true;
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, c4IsPlanted);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, c4IsDropped);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, c4IsDefused);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, c4X);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, c4Y);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, c4Z);
            rowNumber++;
        }
        colNumber = (colNumber + 1) % 6;
    }
    closeMMapFile({fd, stats, file});
}

void ServerState::loadServerState() {
    string generalFileName = "general.csv";
    string generalFilePath = dataPath + "/" + generalFileName;
    string tmpGeneralFileName = "general.csv.tmp.read";
    string tmpGeneralFilePath = dataPath + "/" + tmpGeneralFileName;

    string clientStatesFileName = "state.csv";
    string clientStatesFilePath = dataPath + "/" + clientStatesFileName;
    string tmpClientStatesFileName = "state.csv.tmp.read";
    string tmpClientStatesFilePath = dataPath + "/" + tmpClientStatesFileName;

    string visibilityFileName = "visibility.csv";
    string visibilityFilePath = dataPath + "/" + visibilityFileName;
    string tmpVisibilityFileName = "visibility.csv.tmp.read";
    string tmpVisibilityFilePath = dataPath + "/" + tmpVisibilityFileName;

    string c4FileName = "c4.csv";
    string c4FilePath = dataPath + "/" + c4FileName;
    string tmpC4FileName = "c4.csv.tmp.read";
    string tmpC4FilePath = dataPath + "/" + tmpC4FileName;

    loadTime = std::chrono::system_clock::now();

    bool generalExists = std::filesystem::exists(generalFilePath);
    bool clientStatesExists = std::filesystem::exists(clientStatesFilePath);
    bool visibilityExists = std::filesystem::exists(visibilityFilePath);
    bool c4FileExists = std::filesystem::exists(c4FilePath);
    if (generalExists && clientStatesExists && visibilityExists && c4FileExists) {
        try {
            std::filesystem::rename(generalFilePath, tmpGeneralFilePath);
            std::filesystem::rename(clientStatesFilePath, tmpClientStatesFilePath);
            std::filesystem::rename(visibilityFilePath, tmpVisibilityFilePath);
            std::filesystem::rename(c4FilePath, tmpC4FilePath);
            loadedSuccessfully = true;
        }
        catch(std::filesystem::filesystem_error const& ex) {
            //std::cout
            //        << "what():  " << ex.what() << '\n'
            //        << "path1(): " << ex.path1() << '\n'
            //        << "path2(): " << ex.path2() << '\n'
            //        << "code().value():    " << ex.code().value() << '\n'
            //        << "code().message():  " << ex.code().message() << '\n'
            //        << "code().category(): " << ex.code().category().name() << '\n';
        }
    }
    else {
        if (!generalExists) {
            badPath = generalFilePath;
        }
        else if (!clientStatesExists) {
            badPath = clientStatesFilePath;
        }
        else if (!visibilityExists) {
            badPath = visibilityFilePath;
        }
        else {
            badPath = c4FilePath;
        }
        loadedSuccessfully = false;
        return;
    }

    vector<int64_t> startingPointPerFile = getFileStartingRows({tmpClientStatesFilePath});
    int64_t rows = startingPointPerFile[1];

    clients.resize(rows);
    inputsValid.resize(rows, false);
    
    visibilityClientPairs.clear();

    loadGeneralState(tmpGeneralFilePath);
    loadClientStates(tmpClientStatesFilePath);
    loadVisibilityClientPairs(tmpVisibilityFilePath);
    loadC4State(tmpC4FilePath);

    // build map from server id to CSKnow id
    int maxServerId = -1;
    for (const auto & client : clients) {
        maxServerId = std::max(maxServerId, client.csgoId);
    }
    csgoIdToCSKnowId.resize(maxServerId + 1);
    for (int i = 0; i <= maxServerId; i++) {
        csgoIdToCSKnowId[i] = INVALID_ID;
    }
    for (int i = 0; i < (int) clients.size(); i++) {
        csgoIdToCSKnowId[clients[i].csgoId] = i;
    }
}

void ServerState::saveBotInputs() {
    string inputsFileName = "input.csv";
    string inputsFilePath = dataPath + "/" + inputsFileName;
    string tmpInputsFileName = "input.csv.tmp.write";
    string tmpInputsFilePath = dataPath + "/" + tmpInputsFileName;

    std::stringstream inputsStream;
    inputsStream << "Player Index,Buttons,Input Angle Delta Pct Pitch,Input Angle Delta Pct Yaw\n";

    for (int i = 0; i < (int) inputsValid.size(); i++) {
        if (i < (int) clients.size() && inputsValid[i]) {
            inputsStream << clients[i].csgoId << ","
                << clients[i].buttons << ","
                // FLIPPING TO MATCH YAW AND PITCH
                << clients[i].inputAngleDeltaPctY << ","
                << clients[i].inputAngleDeltaPctX << "\n";
        }
    }

    inputsLog = inputsStream.str();

    std::ofstream fsInputs(tmpInputsFilePath);
    fsInputs << inputsLog;
    fsInputs.close();

    try {
        std::filesystem::rename(tmpInputsFilePath, inputsFilePath);
    }
    catch(std::filesystem::filesystem_error const& ex) { }
}

bool ServerState::saveScript(const vector<string>& scriptLines) const {
    string scriptFileName = "script.txt";
    string scriptFilePath = dataPath + "/" + scriptFileName;
    string tmpScriptFileName = "script.txt.tmp.write";
    string tmpScriptFilePath = dataPath + "/" + tmpScriptFileName;

    if (std::filesystem::exists(scriptFilePath)) {
        // don't save script until prior one is done being processed
        return false;
    }

    std::stringstream scriptStream;

    for (const auto & scriptLine : scriptLines) {
        scriptStream << scriptLine << std::endl;
    }

    std::ofstream fsInputs(tmpScriptFilePath);
    fsInputs << scriptStream.str();
    fsInputs.close();

    try  {
        std::filesystem::rename(tmpScriptFilePath, scriptFilePath);
    }
    catch(std::filesystem::filesystem_error const& ex) { }
    return true;
}

Vec3 ServerState::getC4Pos() const {
    if (c4Exists) {
        return {c4X, c4Y, c4Z};
    }
    else {
        for (const auto & client : clients) {
            if (client.hasC4) {
                return {client.lastEyePosX, client.lastEyePosY, client.lastFootPosZ};
            }
        }
        return {INVALID_ID, INVALID_ID, INVALID_ID};
    }
}
