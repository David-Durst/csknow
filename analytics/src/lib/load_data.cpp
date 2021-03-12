#include "load_data.h"
#include "csv.hpp"
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <string>
#include <iostream>
using std::to_string;
using std::string;

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
               Grenades & grenades, Kills & kills, string dataPath) {
    vector<string> positionPaths = getFilesInDirectory(dataPath + "/position");
//#pragma omp parallel for
    for (int64_t fileIndex = 0; fileIndex < positionPaths.size(); fileIndex++) {
        csv::CSVReader reader(positionPaths[fileIndex]);
        for (const auto & row : reader) {
            position.demoTickNumber.push_back(row["demo tick number"].get<int32_t>());
            position.gameTickNumber.push_back(row["ingame tick"].get<int32_t>());
            position.demoFile.push_back(row["demo file"].get<string>());
            position.matchStarted.push_back(row["match started"].get<bool>());
            position.gamePhase.push_back(row["game phase"].get<int8_t>());
            position.roundsPlayed.push_back(row["rounds played"].get<int8_t>());
            position.isWarmup.push_back(row["is warmup"].get<bool>());
            position.roundStart.push_back(row["round start"].get<bool>());
            position.roundEnd.push_back(row["round end"].get<bool>());
            position.roundEndReason.push_back(row["round end reason"].get<int8_t>());
            position.freezeTimeEnded.push_back(row["freeze time ended"].get<bool>());
            position.tScore.push_back(row["t score"].get<int8_t>());
            position.ctScore.push_back(row["ct score"].get<int8_t>());
            position.numPlayers.push_back(row["num players"].get<int8_t>());

            for (int i = 0; i < NUM_PLAYERS; i++) {
                position.players[i].name.push_back(row["player " + to_string(i) + " name"].get<string>());
                position.players[i].team.push_back(row["player " + to_string(i) + " team"].get<int8_t>());
                position.players[i].xPosition.push_back(row["player " + to_string(i) + " x position"].get<double>());
                position.players[i].yPosition.push_back(row["player " + to_string(i) + " y position"].get<double>());
                position.players[i].zPosition.push_back(row["player " + to_string(i) + " z position"].get<double>());
                position.players[i].xViewDirection.push_back(row["player " + to_string(i) + " x view direction"].get<double>());
                position.players[i].yViewDirection.push_back(row["player " + to_string(i) + " y view direction"].get<double>());
                position.players[i].isAlive.push_back(row["player " + to_string(i) + " is alive"].get<bool>());
                position.players[i].isBlinded.push_back(row["player " + to_string(i) + " is blinded"].get<bool>());

            }
        }
    }
}