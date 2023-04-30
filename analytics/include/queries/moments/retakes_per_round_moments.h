//
// Created by durst on 4/29/23.
//

#ifndef CSKNOW_RETAKES_PER_ROUND_MOMENTS_H
#define CSKNOW_RETAKES_PER_ROUND_MOMENTS_H

#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "load_data.h"
#include "bots/load_save_bot_data.h"
#include "queries/query.h"
#include "geometry.h"
#include "queries/moments/extract_valid_bot_retakes_rounds.h"

namespace csknow::retakes_moments {
    enum RetakeBotType {
        CSKnowLearned = 0,
        CSKnowHeuristic,
        CSGODefault,
        Human
    };
    class RetakesPerRoundMoments : public QueryResult {
    public:
        vector<int64_t> plantTickId;
        vector<int64_t> roundEndTickId;
        vector<int64_t> tickLength;
        vector<int64_t> roundId;
        vector<int64_t> plantId;
        vector<int64_t> defusalId;
        struct TeamMoments {
            // level 0 - base performance
            vector<bool> win;
            // level 1 - key metrics
            vector<double> distanceTraveledPerPlayer;
            vector<double> maxDistanceFromStart;
            vector<double> shotsPerKill;
            vector<double> averageSpeedWhileShooting;
            // level 2 - bug detectors
            vector<int> numPlayersAliveTickBeforeExplosion;
            // base statistics
            vector<RetakeBotType> botType;
            vector<int> numPlayers;
        };
        TeamMoments ctMoments, tMoments;

        RetakesPerRoundMoments() {
            variableLength = true;
            startTickColumn = 0;
            perEventLengthColumn = 2;
        }

        vector<int64_t> filterByForeignKey(int64_t) override { return {}; }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << plantTickId[index] << "," << roundEndTickId[index] << "," << tickLength[index] << ","
              << roundId[index] << "," << plantId[index] << "," << defusalId[index] << ","
              << boolToInt(ctMoments.win[index]) << "," << ctMoments.distanceTraveledPerPlayer[index] << ","
              << ctMoments.maxDistanceFromStart[index] << ","
              << ctMoments.shotsPerKill[index] << "," << ctMoments.averageSpeedWhileShooting[index] << ","
              << ctMoments.numPlayersAliveTickBeforeExplosion[index] << "," << enumAsInt(ctMoments.botType[index]) << ","
              << ctMoments.numPlayers[index] << ","
              << boolToInt(tMoments.win[index]) << "," << tMoments.distanceTraveledPerPlayer[index] << ","
              << tMoments.maxDistanceFromStart[index] << ","
              << tMoments.shotsPerKill[index] << "," << tMoments.averageSpeedWhileShooting[index] << ","
                << tMoments.numPlayersAliveTickBeforeExplosion[index] << "," << enumAsInt(tMoments.botType[index]) << ","
                << tMoments.numPlayers[index];
            s << std::endl;
        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"plant tick id", "round end tick id", "length", "round id", "plant id", "defusal id"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"ct win", "ct distance traveled per player", "ct max distance from start",
                    "ct shots per kill", "ct average speed while shooting", "ct num players alive before explosion",
                    "ct bot type", "ct num players",
                    "t win", "t distance traveled per player", "t max distance from start",
                    "t shots per kill", "t average speed while shooting", "t num players alive before explosion",
                    "t bot type", "t num players"};
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
            file.createDataSet("/data/ct win", ctMoments.win, hdf5FlatCreateProps);
            file.createDataSet("/data/ct distance traveled per player", ctMoments.distanceTraveledPerPlayer, hdf5FlatCreateProps);
            file.createDataSet("/data/ct max distance from start", ctMoments.maxDistanceFromStart, hdf5FlatCreateProps);
            file.createDataSet("/data/ct shots per kill", ctMoments.shotsPerKill, hdf5FlatCreateProps);
            file.createDataSet("/data/ct average speed while shooting", ctMoments.averageSpeedWhileShooting, hdf5FlatCreateProps);
            file.createDataSet("/data/ct num players alive before explosion", ctMoments.numPlayersAliveTickBeforeExplosion, hdf5FlatCreateProps);
            file.createDataSet("/data/ct num players", ctMoments.numPlayers, hdf5FlatCreateProps);
            file.createDataSet("/data/ct bot type", vectorOfEnumsToVectorOfInts(ctMoments.botType), hdf5FlatCreateProps);
            file.createDataSet("/data/t win", tMoments.win, hdf5FlatCreateProps);
            file.createDataSet("/data/t distance traveled per player", tMoments.distanceTraveledPerPlayer, hdf5FlatCreateProps);
            file.createDataSet("/data/t max distance from start", tMoments.maxDistanceFromStart, hdf5FlatCreateProps);
            file.createDataSet("/data/t shots per kill", tMoments.shotsPerKill, hdf5FlatCreateProps);
            file.createDataSet("/data/t average speed while shooting", tMoments.averageSpeedWhileShooting, hdf5FlatCreateProps);
            file.createDataSet("/data/t num players alive before explosion", tMoments.numPlayersAliveTickBeforeExplosion, hdf5FlatCreateProps);
            file.createDataSet("/data/t num players", tMoments.numPlayers, hdf5FlatCreateProps);
            file.createDataSet("/data/t bot type", vectorOfEnumsToVectorOfInts(tMoments.botType), hdf5FlatCreateProps);
        }

        void runQuery(const Games & games, const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                      const WeaponFire & weaponFire, const Kills & kills, const Plants & plants, const Defusals & defusals,
                      const csknow::round_extractor::ExtractValidBotRetakesRounds & extractValidBotRetakesRounds);
    };
}

#endif //CSKNOW_RETAKES_PER_ROUND_MOMENTS_H
