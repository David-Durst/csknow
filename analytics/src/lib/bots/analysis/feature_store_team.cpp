//
// Created by durst on 4/6/23.
//

#include "bots/analysis/feature_store_team.h"
#include "queries/lookback.h"
#include "circular_buffer.h"
#include "file_helpers.h"
#include <atomic>

namespace csknow::feature_store {
    void TeamFeatureStoreResult::init(size_t size) {
        roundId.resize(size, INVALID_ID);
        tickId.resize(size, INVALID_ID);
        valid.resize(size, false);
        c4Status.resize(size, C4Status::NotPlanted);
        c4DistanceToASite.resize(size, INVALID_ID);
        c4DistanceToBSite.resize(size, INVALID_ID);
        for (int j = 0; j < num_orders_per_site; j++) {
            c4DistanceToNearestAOrderNavArea[j].resize(size, INVALID_ID);
            c4DistanceToNearestBOrderNavArea[j].resize(size, INVALID_ID);
        }
        for (int i = 0; i < maxEnemies; i++) {
            columnTData[i].playerId.resize(size, INVALID_ID);
            columnTData[i].distanceToASite.resize(size, 2 * maxWorldDistance);
            columnTData[i].distanceToBSite.resize(size, 2 * maxWorldDistance);
            columnCTData[i].playerId.resize(size, INVALID_ID);
            columnCTData[i].distanceToASite.resize(size, 2 * maxWorldDistance);
            columnCTData[i].distanceToBSite.resize(size, 2 * maxWorldDistance);
            for (int j = 0; j < num_orders_per_site; j++) {
                columnTData[i].distanceToNearestAOrderNavArea[j].resize(size, 2 * maxWorldDistance);
                columnTData[i].distanceToNearestBOrderNavArea[j].resize(size, 2 * maxWorldDistance);
                columnTData[i].distributionNearestAOrders15s[j].resize(size, INVALID_ID);
                columnTData[i].distributionNearestBOrders15s[j].resize(size, INVALID_ID);
                columnTData[i].distributionNearestAOrders30s[j].resize(size, INVALID_ID);
                columnTData[i].distributionNearestBOrders30s[j].resize(size, INVALID_ID);
                columnCTData[i].distanceToNearestAOrderNavArea[j].resize(size, INVALID_ID);
                columnCTData[i].distanceToNearestBOrderNavArea[j].resize(size, INVALID_ID);
                columnCTData[i].distributionNearestAOrders15s[j].resize(size, INVALID_ID);
                columnCTData[i].distributionNearestBOrders15s[j].resize(size, INVALID_ID);
                columnCTData[i].distributionNearestAOrders30s[j].resize(size, INVALID_ID);
                columnCTData[i].distributionNearestBOrders30s[j].resize(size, INVALID_ID);
            }
        }
        this->size = size;
    }

    void TeamFeatureStoreResult::setOrders(const std::vector<csknow::orders::QueryOrder> &orders) {
        aOrders.clear();
        bOrders.clear();
        for (const auto & order : orders) {
            if (order.orderType == orders::OrderType::AOrder) {
                aOrders.push_back(order);
            }
            else {
                bOrders.push_back(order);
            }
        }
    }

    TeamFeatureStoreResult::TeamFeatureStoreResult(size_t size, const std::vector<csknow::orders::QueryOrder> & orders) {
        init(size);
        setOrders(orders);
    }

    void TeamFeatureStoreResult::commitTeamRow(FeatureStorePreCommitBuffer & buffer,
                                               DistanceToPlacesResult & distanceToPlaces,
                                               const nav_mesh::nav_file & navFile,
                                               int64_t roundIndex, int64_t tickIndex) {
        roundId[tickIndex] = roundIndex;
        tickId[tickIndex] = tickIndex;
        valid[tickIndex] = true;

        if (buffer.c4MapData.c4Planted) {
            double c4DistanceToASite =
                distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaId, a_site, navFile);
            double c4DistanceToBSite =
                distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaId, b_site, navFile);
            c4Status[tickIndex] = c4DistanceToASite < c4DistanceToBSite ? C4Status::PlantedA : C4Status::PlantedB;
        }
        else {
            c4Status[tickIndex] = C4Status::NotPlanted;
        }
        c4DistanceToASite[tickIndex] =
            distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaId, a_site, navFile);
        c4DistanceToBSite[tickIndex] =
            distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaId, b_site, navFile);
        for (size_t j = 0; j < num_orders_per_site; j++) {
            double & aOrderDistance = c4DistanceToNearestAOrderNavArea[j][tickIndex];
            aOrderDistance = std::numeric_limits<double>::max();
            for (size_t k = 1; k < aOrders[j].places.size(); k++) {
                aOrderDistance = std::min(aOrderDistance,
                                          distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaIndex, aOrders[j].places[k]));
            }
            double & bOrderDistance = c4DistanceToNearestBOrderNavArea[j][tickIndex];
            bOrderDistance = std::numeric_limits<double>::max();
            for (size_t k = 1; k < bOrders[j].places.size(); k++) {
                bOrderDistance = std::min(bOrderDistance,
                                          distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaIndex, bOrders[j].places[k]));
            }
        }

        /*
        Vec3 t0Pos;
        int64_t areaIndex;
        AreaId areaId;
         */
        for (size_t i = 0; i < buffer.btTeamPlayerData.size(); i++) {
            const BTTeamPlayerData & btTeamPlayerData = buffer.btTeamPlayerData[i];
            auto & columnData = btTeamPlayerData.teamId == ENGINE_TEAM_T ? columnTData : columnCTData;
            size_t columnIndex = btTeamPlayerData.teamId == ENGINE_TEAM_T ?
                buffer.tPlayerIdToIndex[btTeamPlayerData.playerId] : buffer.ctPlayerIdToIndex[btTeamPlayerData.playerId];

            /*
            if (columnIndex == 0 && btTeamPlayerData.teamId == ENGINE_TEAM_T) {
                t0Pos = btTeamPlayerData.curPos;
                areaIndex = btTeamPlayerData.curAreaIndex;
                areaId = btTeamPlayerData.curArea;
            }
             */

            columnData[columnIndex].playerId[tickIndex] = btTeamPlayerData.playerId;
            columnData[columnIndex].distanceToASite[tickIndex] =
                distanceToPlaces.getClosestDistance(btTeamPlayerData.curArea, a_site, navFile);
            columnData[columnIndex].distanceToBSite[tickIndex] =
                distanceToPlaces.getClosestDistance(btTeamPlayerData.curArea, b_site, navFile);
            for (size_t j = 0; j < num_orders_per_site; j++) {
                double & aOrderDistance = columnData[columnIndex].distanceToNearestAOrderNavArea[j][tickIndex];
                aOrderDistance = std::numeric_limits<double>::max();
                // start at 1 so skip tspawn (as all T's spend a significant time pre unfreeze in all orders then)
                for (size_t k = 1; k < aOrders[j].places.size(); k++) {
                    aOrderDistance = std::min(aOrderDistance,
                                              distanceToPlaces.getClosestDistance(btTeamPlayerData.curAreaIndex, aOrders[j].places[k]));
                }
                double & bOrderDistance = columnData[columnIndex].distanceToNearestBOrderNavArea[j][tickIndex];
                bOrderDistance = std::numeric_limits<double>::max();
                for (size_t k = 1; k < bOrders[j].places.size(); k++) {
                    bOrderDistance = std::min(bOrderDistance,
                                              distanceToPlaces.getClosestDistance(btTeamPlayerData.curAreaIndex, bOrders[j].places[k]));
                }
            }
        }

        /*
        if (tickIndex == 1195) {
            std::cout << "T 0 player id " << columnTData[0].playerId[tickIndex] << " pos " << t0Pos.toCSV() << " area id " << areaId
                << " distance to BSite order 2 " << columnTData[0].distanceToNearestBOrderNavArea[0][tickIndex] << std::endl;
            for (size_t k = 0; k < bOrders[2].places.size(); k++) {
                std::cout << "place " << bOrders[2].places[k] << " " << distanceToPlaces.places[bOrders[2].places[k]]
                    << " distance " << distanceToPlaces.getClosestDistance(areaIndex, bOrders[2].places[k]) << std::endl;
            }
        }
         */
    }

    void TeamFeatureStoreResult::computeTeamTickACausalLabels(int64_t curTick, CircularBuffer<int64_t> & futureTracker,
                                                              array<ColumnPlayerData, maxEnemies> & columnData,
                                                              bool future15s) {
        for (size_t playerColumn = 0; playerColumn < maxEnemies; playerColumn++) {
            if (columnData[playerColumn].playerId[curTick] == INVALID_ID) {
                continue;
            }
            // clear out values for current tick
            for (size_t orderPerSite = 0; orderPerSite < num_orders_per_site; orderPerSite++) {
                if (future15s) {
                    columnData[playerColumn].distributionNearestAOrders15s[orderPerSite][curTick] = 0;
                    columnData[playerColumn].distributionNearestBOrders15s[orderPerSite][curTick] = 0;
                }
                else {
                    columnData[playerColumn].distributionNearestAOrders30s[orderPerSite][curTick] = 0;
                    columnData[playerColumn].distributionNearestBOrders30s[orderPerSite][curTick] = 0;
                }
            }
            // want all points where alive, and accounting for ties
            size_t numPointsInDistribution = 0;
            for (int64_t futureTickIndex = 0; futureTickIndex < futureTracker.getCurSize(); futureTickIndex++) {
                int64_t futureTick = futureTracker.fromOldest(futureTickIndex);
                if (futureTick != curTick &&
                    columnData[playerColumn].playerId[curTick] == columnData[playerColumn].playerId[futureTick]) {
                    vector<int64_t> minDistanceAOrders{}, minDistanceBOrders{};
                    double minDistance = std::numeric_limits<double>::max();
                    for (int64_t orderPerSite = 0; orderPerSite < num_orders_per_site; orderPerSite++) {
                        double curADistance = columnData[playerColumn].distanceToNearestAOrderNavArea[orderPerSite][futureTick];
                        if (curADistance < minDistance) {
                            minDistanceAOrders.clear();
                            minDistanceBOrders.clear();
                            minDistanceAOrders.push_back(orderPerSite);
                            minDistance = curADistance;
                        }
                        else if (curADistance == minDistance) {
                            minDistanceAOrders.push_back(orderPerSite);
                        }
                        double curBDistance = columnData[playerColumn].distanceToNearestBOrderNavArea[orderPerSite][futureTick];
                        if (curBDistance < minDistance) {
                            minDistanceAOrders.clear();
                            minDistanceBOrders.clear();
                            minDistanceBOrders.push_back(orderPerSite);
                            minDistance = curBDistance;
                        }
                        else if (curBDistance == minDistance) {
                            minDistanceBOrders.push_back(orderPerSite);
                        }
                    }
                    numPointsInDistribution += minDistanceAOrders.size() + minDistanceBOrders.size();
                    for (const auto & orderPerSite : minDistanceAOrders) {
                        if (future15s) {
                            columnData[playerColumn].distributionNearestAOrders15s[orderPerSite][curTick]++;
                        }
                        else {
                            columnData[playerColumn].distributionNearestAOrders30s[orderPerSite][curTick]++;
                        }
                    }
                    for (const auto & orderPerSite : minDistanceBOrders) {
                        if (future15s) {
                            columnData[playerColumn].distributionNearestBOrders15s[orderPerSite][curTick]++;
                        }
                        else {
                            columnData[playerColumn].distributionNearestBOrders30s[orderPerSite][curTick]++;
                        }
                    }
                }
            }
            if (numPointsInDistribution != 0) {
                for (size_t orderPerSite = 0; orderPerSite < num_orders_per_site; orderPerSite++) {
                    if (future15s) {
                        columnData[playerColumn].distributionNearestAOrders15s[orderPerSite][curTick] /= numPointsInDistribution;
                        columnData[playerColumn].distributionNearestBOrders15s[orderPerSite][curTick] /= numPointsInDistribution;
                    }
                    else {
                        columnData[playerColumn].distributionNearestAOrders30s[orderPerSite][curTick] /= numPointsInDistribution;
                        columnData[playerColumn].distributionNearestBOrders30s[orderPerSite][curTick] /= numPointsInDistribution;
                    }
                }
            }
        }
    }

    void TeamFeatureStoreResult::computeAcausalLabels(const Games & games, const Rounds & rounds,
                                                      const Ticks & ticks) {
        std::atomic<int64_t> roundsProcessed = 0;
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            TickRates tickRates = computeTickRates(games, rounds, roundIndex);
            CircularBuffer<int64_t> ticks15sFutureTracker(15), ticks30sFutureTracker(30);
            for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].maxId;
                 tickIndex >= rounds.ticksPerRound[roundIndex].minId; tickIndex--) {
                // add a new tick every second
                if (ticks15sFutureTracker.isEmpty() ||
                    secondsBetweenTicks(ticks, tickRates, tickIndex, ticks15sFutureTracker.fromNewest()) >= 1.) {
                    ticks15sFutureTracker.enqueue(tickIndex);
                }
                if (ticks30sFutureTracker.isEmpty() ||
                    secondsBetweenTicks(ticks, tickRates, tickIndex, ticks30sFutureTracker.fromNewest()) >= 1.) {
                    ticks30sFutureTracker.enqueue(tickIndex);
                }
                computeTeamTickACausalLabels(tickIndex, ticks15sFutureTracker, columnCTData, true);
                computeTeamTickACausalLabels(tickIndex, ticks30sFutureTracker, columnCTData, false);
                computeTeamTickACausalLabels(tickIndex, ticks15sFutureTracker, columnTData, true);
                computeTeamTickACausalLabels(tickIndex, ticks30sFutureTracker, columnTData, false);
                /*
                if (tickIndex == 1195) {
                    std::cout << "cur tick " << tickIndex << " 30s tick " << ticks30sFutureTracker.fromOldest() << std::endl;
                    std::cout << "distribution nearest a order 0 30s CT 0 " << columnCTData[0].distributionNearestAOrders30s[0][tickIndex] << std::endl;
                    std::cout << "distribution nearest a order 1 30s CT 0 " << columnCTData[0].distributionNearestAOrders30s[1][tickIndex] << std::endl;
                    std::cout << "distribution nearest a order 2 30s CT 0 " << columnCTData[0].distributionNearestAOrders30s[2][tickIndex] << std::endl;
                    std::cout << "distribution nearest b order 0 30s CT 0 " << columnCTData[0].distributionNearestBOrders30s[0][tickIndex] << std::endl;
                    std::cout << "distribution nearest b order 1 30s CT 0 " << columnCTData[0].distributionNearestBOrders30s[1][tickIndex] << std::endl;
                    std::cout << "distribution nearest b order 2 30s CT 0 " << columnCTData[0].distributionNearestBOrders30s[2][tickIndex] << std::endl;
                }
                 */
            }
            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
        }
    }

    void TeamFeatureStoreResult::toHDF5Inner(HighFive::File & file) {
        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        hdf5FlatCreateProps.add(HighFive::Deflate(6));
        hdf5FlatCreateProps.add(HighFive::Chunking(roundId.size()));

        file.createDataSet("/data/round id", roundId, hdf5FlatCreateProps);
        file.createDataSet("/data/tick id", tickId, hdf5FlatCreateProps);
        file.createDataSet("/data/valid", valid, hdf5FlatCreateProps);
        file.createDataSet("/data/c4 status", vectorOfEnumsToVectorOfInts(c4Status), hdf5FlatCreateProps);
        file.createDataSet("/data/c4 distance to a site", c4DistanceToASite, hdf5FlatCreateProps);
        file.createDataSet("/data/c4 distance to b site", c4DistanceToBSite, hdf5FlatCreateProps);
        for (size_t orderIndex = 0; orderIndex < num_orders_per_site; orderIndex++) {
            string orderIndexStr = std::to_string(orderIndex);
            file.createDataSet("/data/c4 distance to nearest a order " + orderIndexStr + " nav area", c4DistanceToNearestAOrderNavArea[orderIndex], hdf5FlatCreateProps);
            file.createDataSet("/data/c4 distance to nearest b order " + orderIndexStr + " nav area", c4DistanceToNearestBOrderNavArea[orderIndex], hdf5FlatCreateProps);
        }
        for (size_t columnDataIndex = 0; columnDataIndex < getAllColumnData().size(); columnDataIndex++) {
            const array<ColumnPlayerData, maxEnemies> & columnData = getAllColumnData()[columnDataIndex];
            string columnTeam = allColumnDataTeam[columnDataIndex];
            for (size_t columnPlayer = 0; columnPlayer < columnData.size(); columnPlayer++) {
                string iStr = std::to_string(columnPlayer);
                file.createDataSet("/data/player id " + columnTeam + " " + iStr,
                                   columnData[columnPlayer].playerId, hdf5FlatCreateProps);
                file.createDataSet("/data/distance to a site " + columnTeam + " " + iStr,
                                   columnData[columnPlayer].distanceToASite, hdf5FlatCreateProps);
                file.createDataSet("/data/distance to b site " + columnTeam + " " + iStr,
                                   columnData[columnPlayer].distanceToBSite, hdf5FlatCreateProps);
                for (size_t orderIndex = 0; orderIndex < num_orders_per_site; orderIndex++) {
                    string orderIndexStr = std::to_string(orderIndex);
                    file.createDataSet("/data/distance to nearest a order " + orderIndexStr + " nav area " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].distanceToNearestAOrderNavArea[orderIndex], hdf5FlatCreateProps);
                    file.createDataSet("/data/distance to nearest b order " + orderIndexStr + " nav area " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].distanceToNearestBOrderNavArea[orderIndex], hdf5FlatCreateProps);
                    file.createDataSet("/data/distribution nearest a order " + orderIndexStr + " 15s " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].distributionNearestAOrders15s[orderIndex], hdf5FlatCreateProps);
                    file.createDataSet("/data/distribution nearest b order " + orderIndexStr + " 15s " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].distributionNearestBOrders15s[orderIndex], hdf5FlatCreateProps);
                    file.createDataSet("/data/distribution nearest a order " + orderIndexStr + " 30s " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].distributionNearestAOrders30s[orderIndex], hdf5FlatCreateProps);
                    file.createDataSet("/data/distribution nearest b order " + orderIndexStr + " 30s " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].distributionNearestBOrders30s[orderIndex], hdf5FlatCreateProps);
                }
            }
        }
    }
}
