#include "load_data.h"
#include "csv.hpp"
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <string>
#include <iostream>
#include <atomic>

using std::to_string;
using std::string;

void printProgress(double progress) {
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

const string placeholderFileName = ".placeholder";
vector<string> getFilesInDirectory(string path) {
    vector<string> result;
    DIR * dir;
    struct dirent * en;
    dir = opendir(path.c_str());
    if (dir) {
        while ((en = readdir(dir)) != NULL) {
            if (en->d_type == DT_REG && placeholderFileName.compare(en->d_name) != 0) {
                result.push_back(path + "/" + en->d_name);
            }
        }
        closedir(dir);
    }
    return result;
}

void loadData(Position & position, Spotted & spotted, WeaponFire & weaponFire, PlayerHurt & playerHurt,
               Grenades & grenades, Kills & kills, string dataPath, OpenFiles & openFiles) {
    vector<string> positionPaths = getFilesInDirectory(dataPath + "/position");
    vector<PositionBuilder> positions{positionPaths.size()};
    vector<int64_t> startingPointPerFile;
    std::cout << "loading positions off disk" << std::endl;
    std::atomic<int64_t> filesProcessed = 0;
    openFiles.paths.clear();
    #pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < positionPaths.size(); fileIndex++) {
        try {
            openFiles.paths.insert(positionPaths[fileIndex]);
            std::cerr << "open files: ";
            for (const auto & path : openFiles.paths) {
                std::cerr << path << "," << std::endl;
            }
            std::cerr << std::endl;
            csv::CSVReader reader(positionPaths[fileIndex]);
            for (const auto & row : reader) {
                positions[fileIndex].demoTickNumber.push_back(row["demo tick number"].get<int32_t>());
                positions[fileIndex].gameTickNumber.push_back(row["ingame tick"].get<int32_t>());
                positions[fileIndex].demoFile.push_back(row["demo file"].get<string>());
                positions[fileIndex].matchStarted.push_back(row["match started"].get<bool>());
                positions[fileIndex].gamePhase.push_back(row["game phase"].get<int8_t>());
                positions[fileIndex].roundsPlayed.push_back(row["rounds played"].get<int8_t>());
                positions[fileIndex].isWarmup.push_back(row["is warmup"].get<bool>());
                positions[fileIndex].roundStart.push_back(row["round start"].get<bool>());
                positions[fileIndex].roundEnd.push_back(row["round end"].get<bool>());
                positions[fileIndex].roundEndReason.push_back(row["round end reason"].get<int8_t>());
                positions[fileIndex].freezeTimeEnded.push_back(row["freeze time ended"].get<bool>());
                positions[fileIndex].tScore.push_back(row["t score"].get<int8_t>());
                positions[fileIndex].ctScore.push_back(row["ct score"].get<int8_t>());
                positions[fileIndex].numPlayers.push_back(row["num players"].get<int8_t>());

                for (int i = 0; i < NUM_PLAYERS; i++) {
                    positions[fileIndex].players[i].name.push_back(row["player " + to_string(i) + " name"].get<string>());
                    positions[fileIndex].players[i].team.push_back(row["player " + to_string(i) + " team"].get<int8_t>());
                    positions[fileIndex].players[i].xPosition.push_back(row["player " + to_string(i) + " x position"].get<double>());
                    positions[fileIndex].players[i].yPosition.push_back(row["player " + to_string(i) + " y position"].get<double>());
                    positions[fileIndex].players[i].zPosition.push_back(row["player " + to_string(i) + " z position"].get<double>());
                    positions[fileIndex].players[i].xViewDirection.push_back(row["player " + to_string(i) + " x view direction"].get<double>());
                    positions[fileIndex].players[i].yViewDirection.push_back(row["player " + to_string(i) + " y view direction"].get<double>());
                    positions[fileIndex].players[i].isAlive.push_back(row["player " + to_string(i) + " is alive"].get<bool>());
                    positions[fileIndex].players[i].isBlinded.push_back(row["player " + to_string(i) + " is blinded"].get<bool>());

                }
            }
        }
        catch (const std::exception &exc){
            std::cerr << "problem with file " << positionPaths[fileIndex] << std::endl;
            std::cerr << exc.what();
        }
        openFiles.paths.erase(positionPaths[fileIndex]);
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / positionPaths.size());
    }
    std::cout << std::endl;

    std::cout << "allocating vectors" << std::endl;
    startingPointPerFile.push_back(0);
    for (int64_t fileIndex = 0; fileIndex < positionPaths.size(); fileIndex++) {
        startingPointPerFile.push_back(startingPointPerFile[fileIndex] + positions[fileIndex].demoTickNumber.size());
    }
    int64_t rows = startingPointPerFile[startingPointPerFile.size() - 1];
    position.size = rows;
    position.demoTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
    position.gameTickNumber = (int32_t *) malloc(rows * sizeof(int32_t));
    position.demoFile = (string *) malloc(rows * sizeof(string));
    position.matchStarted = (bool *) malloc(rows * sizeof(bool));
    position.gamePhase = (int8_t *) malloc(rows * sizeof(int8_t));
    position.roundsPlayed = (int8_t *) malloc(rows * sizeof(int8_t));
    position.isWarmup = (bool *) malloc(rows * sizeof(bool));
    position.roundStart = (bool *) malloc(rows * sizeof(bool));
    position.roundEnd = (bool *) malloc(rows * sizeof(bool));
    position.roundEndReason = (int8_t *) malloc(rows * sizeof(int8_t));
    position.freezeTimeEnded = (bool *) malloc(rows * sizeof(bool));
    position.tScore = (int8_t *) malloc(rows * sizeof(int8_t));
    position.ctScore = (int8_t *) malloc(rows * sizeof(int8_t));
    position.numPlayers = (int8_t *) malloc(rows * sizeof(int8_t));
    for (int i = 0; i < NUM_PLAYERS; i++) {
        position.players[i].name = (string *) malloc(rows * sizeof(string));
        position.players[i].team = (int8_t *) malloc(rows * sizeof(int8_t));
        position.players[i].xPosition = (double *) malloc(rows * sizeof(double));
        position.players[i].yPosition = (double *) malloc(rows * sizeof(double));
        position.players[i].zPosition = (double *) malloc(rows * sizeof(double));
        position.players[i].xViewDirection = (double *) malloc(rows * sizeof(double));
        position.players[i].yViewDirection = (double *) malloc(rows * sizeof(double));
        position.players[i].isAlive = (bool *) malloc(rows * sizeof(bool));
        position.players[i].isBlinded = (bool *) malloc(rows * sizeof(bool));
    }

    std::cout << "merging vectors" << std::endl;
    filesProcessed = 0;
    #pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < positionPaths.size(); fileIndex++) {
        int64_t fileRow = 0;
        for (int64_t positionIndex = startingPointPerFile[fileIndex]; positionIndex < startingPointPerFile[fileIndex-1];
            positionIndex++) {
            fileRow++;
            position.demoTickNumber[positionIndex] = positions[fileIndex].demoTickNumber[fileRow];
            position.gameTickNumber[positionIndex] = positions[fileIndex].gameTickNumber[fileRow];
            position.demoFile[positionIndex] = positions[fileIndex].demoFile[fileRow];
            position.matchStarted[positionIndex] = positions[fileIndex].matchStarted[fileRow];
            position.gamePhase[positionIndex] = positions[fileIndex].gamePhase[fileRow];
            position.roundsPlayed[positionIndex] = positions[fileIndex].roundsPlayed[fileRow];
            position.isWarmup[positionIndex] = positions[fileIndex].isWarmup[fileRow];
            position.roundStart[positionIndex] = positions[fileIndex].roundStart[fileRow];
            position.roundEnd[positionIndex] = positions[fileIndex].roundEnd[fileRow];
            position.roundEndReason[positionIndex] = positions[fileIndex].roundEndReason[fileRow];
            position.freezeTimeEnded[positionIndex] = positions[fileIndex].freezeTimeEnded[fileRow];
            position.tScore[positionIndex] = positions[fileIndex].tScore[fileRow];
            position.ctScore[positionIndex] = positions[fileIndex].ctScore[fileRow];
            position.numPlayers[positionIndex] = positions[fileIndex].numPlayers[fileRow];
            for (int i = 0; i < NUM_PLAYERS; i++) {
                position.players[i].name[positionIndex] = positions[fileIndex].players[i].name[fileRow];
                position.players[i].team[positionIndex] = positions[fileIndex].players[i].team[fileRow];
                position.players[i].xPosition[positionIndex] = positions[fileIndex].players[i].xPosition[fileRow];
                position.players[i].yPosition[positionIndex] = positions[fileIndex].players[i].yPosition[fileRow];
                position.players[i].zPosition[positionIndex] = positions[fileIndex].players[i].zPosition[fileRow];
                position.players[i].xViewDirection[positionIndex] = positions[fileIndex].players[i].xViewDirection[fileRow];
                position.players[i].yViewDirection[positionIndex] = positions[fileIndex].players[i].yViewDirection[fileRow];
                position.players[i].isAlive[positionIndex] = positions[fileIndex].players[i].isAlive[fileRow];
                position.players[i].isBlinded[positionIndex] = positions[fileIndex].players[i].isBlinded[fileRow];
            }
        }
        filesProcessed++;
        printProgress((filesProcessed * 1.0) / positionPaths.size());
    }
    std::cout << std::endl;
}