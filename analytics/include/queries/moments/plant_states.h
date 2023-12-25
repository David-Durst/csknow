//
// Created by durst on 4/27/23.
//

#ifndef CSKNOW_PLANT_STATES_H
#define CSKNOW_PLANT_STATES_H

#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "load_data.h"
#include "queries/query.h"
#include "geometry.h"
#include "enum_helpers.h"
#include "bots/analysis/feature_store.h"

namespace csknow::plant_states {
    constexpr int max_players_per_team = 5;

    class PlantStatesResult : public QueryResult {
    public:
        vector<RangeIndexEntry> rowIndicesPerRound;
        vector<int64_t> plantTickId;
        vector<int64_t> roundEndTickId;
        vector<int64_t> tickLength;
        vector<int64_t> roundId;
        vector<int64_t> plantId;
        vector<int64_t> defusalId;
        vector<Vec3> c4Pos;
        vector<TeamId> winnerTeam;
        vector<bool> c4Defused;
        IntervalIndex plantStatesPerTick;
        struct PlayerState {
            vector<bool> alive;
            vector<Vec3> pos;
            vector<Vec2> viewAngle;
            vector<float> health;
            vector<float> armor;
        };
        array<PlayerState, max_players_per_team> ctPlayerStates, tPlayerStates;
        vector<Vec3> cameraPos;
        vector<Vec2> cameraViewAngle;


        PlantStatesResult() {
            variableLength = true;
            startTickColumn = 0;
            perEventLengthColumn = 2;
        }

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
            vector<int64_t> result;
            // Normally would segfault, but this query is slow so I don't run for all rounds in debug cases
            if (otherTableIndex >= static_cast<int64_t>(rowIndicesPerRound.size())) {
                return result;
            }
            for (int64_t i = rowIndicesPerRound[otherTableIndex].minId; i <= rowIndicesPerRound[otherTableIndex].maxId; i++) {
                if (i == -1) {
                    continue;
                }
                result.push_back(i);
            }
            return result;
        }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << plantTickId[index] << "," << roundEndTickId[index] << "," << tickLength[index] << ","
              << roundId[index] << "," << plantId[index] << "," << defusalId[index] << "," << c4Pos[index].toCSV() << ","
              << winnerTeam[index] << "," << boolToInt(c4Defused[index]);
            s << std::endl;
        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"plant tick id", "round end tick id", "length", "round id", "plant id", "defusal id"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"c4 pos x", "c4 pos y", "c4 pos z", "winner team", "c4 defused"};
        }

        void toHDF5Inner(HighFive::File & file) override {
            HighFive::DataSetCreateProps hdf5FlatCreateProps;
            hdf5FlatCreateProps.add(HighFive::Deflate(6));
            hdf5FlatCreateProps.add(HighFive::Chunking(plantTickId.size()));

            file.createDataSet("/data/plant tick id", plantTickId, hdf5FlatCreateProps);
            file.createDataSet("/data/round end tick id", roundEndTickId, hdf5FlatCreateProps);
            file.createDataSet("/data/tick length", tickLength, hdf5FlatCreateProps);
            file.createDataSet("/data/round id", roundId, hdf5FlatCreateProps);
            file.createDataSet("/data/plant id", plantId, hdf5FlatCreateProps);
            file.createDataSet("/data/defusal id", defusalId, hdf5FlatCreateProps);
            saveVec3VectorToHDF5(c4Pos, file, "c4 pos", hdf5FlatCreateProps);
            file.createDataSet("/data/winner team", winnerTeam, hdf5FlatCreateProps);
            file.createDataSet("/data/c4 defused", c4Defused, hdf5FlatCreateProps);
            for (size_t i = 0; i < max_players_per_team; i++) {
                string iStr = std::to_string(i);
                file.createDataSet("/data/ct " + iStr + " alive", ctPlayerStates[i].alive, hdf5FlatCreateProps);
                saveVec3VectorToHDF5(ctPlayerStates[i].pos, file, "ct " + iStr + " pos", hdf5FlatCreateProps);
                saveVec2VectorToHDF5(ctPlayerStates[i].viewAngle, file, "ct " + iStr + " view angle", hdf5FlatCreateProps);
                file.createDataSet("/data/ct " + iStr + " health", ctPlayerStates[i].health, hdf5FlatCreateProps);
                file.createDataSet("/data/ct " + iStr + " armor", ctPlayerStates[i].armor, hdf5FlatCreateProps);
            }
            for (size_t i = 0; i < max_players_per_team; i++) {
                string iStr = std::to_string(i);
                file.createDataSet("/data/t " + iStr + " alive", tPlayerStates[i].alive, hdf5FlatCreateProps);
                saveVec3VectorToHDF5(tPlayerStates[i].pos, file, "t " + iStr + " pos", hdf5FlatCreateProps);
                saveVec2VectorToHDF5(tPlayerStates[i].viewAngle, file, "t " + iStr + " view angle", hdf5FlatCreateProps);
                file.createDataSet("/data/t " + iStr + " health", tPlayerStates[i].health, hdf5FlatCreateProps);
                file.createDataSet("/data/t " + iStr + " armor", tPlayerStates[i].armor, hdf5FlatCreateProps);
            }
        }

        void load(const string& filePath) {
            // We open the file as read-only:
            HighFive::File file(filePath, HighFive::File::ReadOnly);

            plantTickId = file.getDataSet("/data/plant tick id").read<std::vector<int64_t>>();
            roundEndTickId = file.getDataSet("/data/round end tick id").read<std::vector<int64_t>>();
            tickLength = file.getDataSet("/data/tick length").read<std::vector<int64_t>>();
            roundId = file.getDataSet("/data/round id").read<std::vector<int64_t>>();
            plantId = file.getDataSet("/data/plant id").read<std::vector<int64_t>>();
            defusalId = file.getDataSet("/data/defusal id").read<std::vector<int64_t>>();
            loadVec3VectorFromHDF5(c4Pos, file, "c4 pos");
            winnerTeam = file.getDataSet("/data/winner team").read<std::vector<TeamId>>();
            c4Defused = file.getDataSet("/data/c4 defused").read<std::vector<bool>>();

            for (size_t i = 0; i < max_players_per_team; i++) {
                string iStr = std::to_string(i);
                auto ctPlayerAliveDataset = file.getDataSet("/data/ct " + iStr + " alive");
                ctPlayerStates[i].alive = ctPlayerAliveDataset.read<std::vector<bool>>();
                loadVec3VectorFromHDF5(ctPlayerStates[i].pos, file, "ct " + iStr + " pos");
                loadVec2VectorFromHDF5(ctPlayerStates[i].viewAngle, file, "ct " + iStr + " view angle");
                ctPlayerStates[i].health = file.getDataSet("/data/ct " + iStr + " health").read<std::vector<float>>();
                ctPlayerStates[i].armor = file.getDataSet("/data/ct " + iStr + " armor").read<std::vector<float>>();
            }
            for (size_t i = 0; i < max_players_per_team; i++) {
                string iStr = std::to_string(i);
                auto tPlayerAliveDataset = file.getDataSet("/data/t " + iStr + " alive");
                tPlayerStates[i].alive = tPlayerAliveDataset.read<std::vector<bool>>();
                loadVec3VectorFromHDF5(tPlayerStates[i].pos, file, "t " + iStr + " pos");
                loadVec2VectorFromHDF5(tPlayerStates[i].viewAngle, file, "t " + iStr + " view angle");
                tPlayerStates[i].health = file.getDataSet("/data/t " + iStr + " health").read<std::vector<float>>();
                tPlayerStates[i].armor = file.getDataSet("/data/t " + iStr + " armor").read<std::vector<float>>();
            }

            size = plantTickId.size();
        }

        void loadFromPython(const string& filePath, bool loadCamera) {
            // We open the file as read-only:
            HighFive::File file(filePath, HighFive::File::ReadOnly);

            plantTickId = file.getDataSet("/data/plant tick id").read<std::vector<int64_t>>();
            roundEndTickId = file.getDataSet("/data/round end tick id").read<std::vector<int64_t>>();
            tickLength = file.getDataSet("/data/tick length").read<std::vector<int64_t>>();
            roundId = file.getDataSet("/data/round id").read<std::vector<int64_t>>();
            plantId = file.getDataSet("/data/plant id").read<std::vector<int64_t>>();
            defusalId = file.getDataSet("/data/defusal id").read<std::vector<int64_t>>();
            loadVec3VectorFromHDF5(c4Pos, file, "c4 pos");
            winnerTeam = file.getDataSet("/data/winner team").read<std::vector<TeamId>>();
            c4Defused = file.getDataSet("/data/c4 defused").read<std::vector<bool>>();

            for (size_t i = 0; i < max_players_per_team; i++) {
                string iStr = std::to_string(i);
                ctPlayerStates[i].alive = file.getDataSet("/data/alive CT " + iStr).read<std::vector<bool>>();
                loadVec3VectorFromHDF5(ctPlayerStates[i].pos, file, "player pos CT " + iStr);
                loadVec2VectorFromHDF5(ctPlayerStates[i].viewAngle, file, "player view angle CT " + iStr);
                ctPlayerStates[i].health = file.getDataSet("/data/player health CT " + iStr).read<std::vector<float>>();
                ctPlayerStates[i].armor = file.getDataSet("/data/player armor CT " + iStr).read<std::vector<float>>();
            }
            for (size_t i = 0; i < max_players_per_team; i++) {
                string iStr = std::to_string(i);
                tPlayerStates[i].alive = file.getDataSet("/data/alive T " + iStr).read<std::vector<bool>>();
                loadVec3VectorFromHDF5(tPlayerStates[i].pos, file, "player pos T " + iStr);
                loadVec2VectorFromHDF5(tPlayerStates[i].viewAngle, file, "player view angle T " + iStr);
                tPlayerStates[i].health = file.getDataSet("/data/player health T " + iStr).read<std::vector<float>>();
                tPlayerStates[i].armor = file.getDataSet("/data/player armor T " + iStr).read<std::vector<float>>();
            }

            if (loadCamera) {
                loadVec3VectorFromHDF5(cameraPos, file, "camera pos");
                loadVec2VectorFromHDF5(cameraViewAngle, file, "camera view angle");
            }

            size = plantTickId.size();
        }

        void runQuery(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                      const Plants & plants, const Defusals & defusals);
    };

}


#endif //CSKNOW_PLANT_STATES_H
