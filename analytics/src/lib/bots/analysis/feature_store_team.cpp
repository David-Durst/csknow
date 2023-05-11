//
// Created by durst on 4/6/23.
//

#include "bots/analysis/feature_store_team.h"
#include "circular_buffer.h"
#include "file_helpers.h"
#include <atomic>
#include <cmath>
#include <csignal>

namespace csknow::feature_store {
    void TeamFeatureStoreResult::init(size_t totalSize) {
        size_t size = (totalSize == 1) ? totalSize : totalSize / every_nth_row;
        //size_t size = totalSize;
        roundId.resize(size, INVALID_ID);
        roundNumber.resize(size, INVALID_ID);
        tickId.resize(size, INVALID_ID);
        gameTickNumber.resize(size, INVALID_ID);
        valid.resize(size, false);
        freezeTimeEnded.resize(size, false);
        retakeSaveRoundTick.resize(size, false);
        c4Status.resize(size, C4Status::NotPlanted);
        c4TicksSincePlant.resize(size, INVALID_ID);
        for (int j = 0; j < num_c4_timer_buckets; j++) {
            c4TimerBucketed[j].resize(size, false);
        }
        c4Pos.resize(size, invalidWorldPos);
        c4DistanceToASite.resize(size, INVALID_ID);
        c4DistanceToBSite.resize(size, INVALID_ID);
        for (int j = 0; j < num_orders_per_site; j++) {
            c4DistanceToNearestAOrderNavArea[j].resize(size, INVALID_ID);
            c4DistanceToNearestBOrderNavArea[j].resize(size, INVALID_ID);
        }
        for (int i = 0; i < maxEnemies; i++) {
            for (int j = 0; j < maxEnemies; j++) {
                columnTData[i].indexOnTeam[j].resize(size, j == i);
                columnCTData[i].indexOnTeam[j].resize(size, j == i);
            }
            columnTData[i].playerId.resize(size, INVALID_ID);
            columnTData[i].ctTeam.resize(size, false);
            columnTData[i].footPos.resize(size, invalidWorldPos);
            columnTData[i].velocity.resize(size, {INVALID_ID, INVALID_ID, INVALID_ID});
            columnTData[i].distanceToASite.resize(size, 2 * maxWorldDistance);
            columnTData[i].distanceToBSite.resize(size, 2 * maxWorldDistance);
            columnCTData[i].playerId.resize(size, INVALID_ID);
            columnCTData[i].ctTeam.resize(size, true);
            columnCTData[i].footPos.resize(size, invalidWorldPos);
            columnCTData[i].velocity.resize(size, {INVALID_ID, INVALID_ID, INVALID_ID});
            columnCTData[i].distanceToASite.resize(size, 2 * maxWorldDistance);
            columnCTData[i].distanceToBSite.resize(size, 2 * maxWorldDistance);
            for (int j = 0; j < num_prior_ticks; j++) {
                columnTData[i].priorFootPos[j].resize(size, invalidWorldPos);
                columnCTData[i].priorFootPos[j].resize(size, invalidWorldPos);
            }
            for (int j = 0; j < num_orders_per_site; j++) {
                columnTData[i].distanceToNearestAOrderNavArea[j].resize(size, 2 * maxWorldDistance);
                columnTData[i].distanceToNearestBOrderNavArea[j].resize(size, 2 * maxWorldDistance);
                columnTData[i].distributionNearestAOrders[j].resize(size, INVALID_ID);
                columnTData[i].distributionNearestBOrders[j].resize(size, INVALID_ID);
                //columnTData[i].distributionNearestAOrders15s[j].resize(size, INVALID_ID);
                //columnTData[i].distributionNearestBOrders15s[j].resize(size, INVALID_ID);
                //columnTData[i].distributionNearestAOrders30s[j].resize(size, INVALID_ID);
                //columnTData[i].distributionNearestBOrders30s[j].resize(size, INVALID_ID);
                columnCTData[i].distanceToNearestAOrderNavArea[j].resize(size, INVALID_ID);
                columnCTData[i].distanceToNearestBOrderNavArea[j].resize(size, INVALID_ID);
                columnCTData[i].distributionNearestAOrders[j].resize(size, INVALID_ID);
                columnCTData[i].distributionNearestBOrders[j].resize(size, INVALID_ID);
                //columnCTData[i].distributionNearestAOrders15s[j].resize(size, INVALID_ID);
                //columnCTData[i].distributionNearestBOrders15s[j].resize(size, INVALID_ID);
                //columnCTData[i].distributionNearestAOrders30s[j].resize(size, INVALID_ID);
                //columnCTData[i].distributionNearestBOrders30s[j].resize(size, INVALID_ID);
            }
            for (int j = 0; j < num_places; j++) {
                columnTData[i].curPlace[j].resize(size, false);
                columnTData[i].distributionNearestPlace[j].resize(size, INVALID_ID);
                //columnTData[i].distributionNearestPlace7to15s[j].resize(size, INVALID_ID);
                columnCTData[i].curPlace[j].resize(size, false);
                columnCTData[i].distributionNearestPlace[j].resize(size, INVALID_ID);
                //columnCTData[i].distributionNearestPlace7to15s[j].resize(size, INVALID_ID);
                for (int k = 0; k < num_prior_ticks; k++) {
                    columnTData[i].priorPlaces[k][j].resize(size, false);
                    columnCTData[i].priorPlaces[k][j].resize(size, false);
                }
            }
            for (int j = 0; j < area_grid_size; j++) {
                columnTData[i].areaGridCellInPlace[j].resize(size, false);
                columnTData[i].distributionNearestAreaGridInPlace[j].resize(size, INVALID_ID);
                //columnTData[i].distributionNearestAreaGridInPlace7to15s[j].resize(size, INVALID_ID);
                columnCTData[i].areaGridCellInPlace[j].resize(size, false);
                columnCTData[i].distributionNearestAreaGridInPlace[j].resize(size, INVALID_ID);
                //columnCTData[i].distributionNearestAreaGridInPlace7to15s[j].resize(size, INVALID_ID);
                for (int k = 0; k < num_prior_ticks; k++) {
                    columnTData[i].priorAreaGridCellInPlace[k][j].resize(size, false);
                    columnCTData[i].priorAreaGridCellInPlace[k][j].resize(size, false);
                }
            }
            for (int j = 0; j < delta_pos_grid_num_cells; j++) {
                columnTData[i].deltaPos[j].resize(size, false);
                columnCTData[i].deltaPos[j].resize(size, false);
            }
        }
        this->size = size;
        //checkPossiblyBadValue();
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
    
    void TeamFeatureStoreResult::reinit() {
        for (int64_t rowIndex = 0; rowIndex < size; rowIndex++) {
            roundId[rowIndex] = INVALID_ID;
            roundNumber[rowIndex] = INVALID_ID;
            tickId[rowIndex] = INVALID_ID;
            gameTickNumber[rowIndex] = INVALID_ID;
            valid[rowIndex] = false;
            freezeTimeEnded[rowIndex] = false;
            retakeSaveRoundTick[rowIndex] = false;
            c4Status[rowIndex] = C4Status::NotPlanted;
            c4TicksSincePlant[rowIndex] = INVALID_ID;
            for (int j = 0; j < num_c4_timer_buckets; j++) {
                c4TimerBucketed[j][rowIndex] = INVALID_ID;
            }
            c4Pos[rowIndex] = invalidWorldPos;
            c4DistanceToASite[rowIndex] = INVALID_ID;
            c4DistanceToBSite[rowIndex] = INVALID_ID;
            for (int j = 0; j < num_orders_per_site; j++) {
                c4DistanceToNearestAOrderNavArea[j][rowIndex] = INVALID_ID;
                c4DistanceToNearestBOrderNavArea[j][rowIndex] = INVALID_ID;
            }
            for (int i = 0; i < maxEnemies; i++) {
                for (int j = 0; j < maxEnemies; j++) {
                    columnTData[i].indexOnTeam[j][rowIndex] = j == i;
                    columnCTData[i].indexOnTeam[j][rowIndex] = j == i;
                }
                columnTData[i].playerId[rowIndex] = INVALID_ID;
                columnTData[i].ctTeam[rowIndex] = false;
                columnTData[i].footPos[rowIndex] = invalidWorldPos;
                columnTData[i].velocity[rowIndex] = {INVALID_ID, INVALID_ID, INVALID_ID};
                columnTData[i].distanceToASite[rowIndex] = 2 * maxWorldDistance;
                columnTData[i].distanceToBSite[rowIndex] = 2 * maxWorldDistance;
                columnCTData[i].playerId[rowIndex] = INVALID_ID;
                columnTData[i].ctTeam[rowIndex] = true;
                columnCTData[i].footPos[rowIndex] = invalidWorldPos;
                columnCTData[i].velocity[rowIndex] = {INVALID_ID, INVALID_ID, INVALID_ID};
                columnCTData[i].distanceToASite[rowIndex] = 2 * maxWorldDistance;
                columnCTData[i].distanceToBSite[rowIndex] = 2 * maxWorldDistance;
                for (int j = 0; j < num_prior_ticks; j++) {
                    columnTData[i].priorFootPos[j][rowIndex] = invalidWorldPos;
                    columnCTData[i].priorFootPos[j][rowIndex] = invalidWorldPos;
                }
                for (int j = 0; j < num_orders_per_site; j++) {
                    columnTData[i].distanceToNearestAOrderNavArea[j][rowIndex] = 2 * maxWorldDistance;
                    columnTData[i].distanceToNearestBOrderNavArea[j][rowIndex] = 2 * maxWorldDistance;
                    columnTData[i].distributionNearestAOrders[j][rowIndex] = INVALID_ID;
                    columnTData[i].distributionNearestBOrders[j][rowIndex] = INVALID_ID;
                    //columnTData[i].distributionNearestAOrders15s[j][rowIndex] = INVALID_ID;
                    //columnTData[i].distributionNearestBOrders15s[j][rowIndex] = INVALID_ID;
                    //columnTData[i].distributionNearestAOrders30s[j][rowIndex] = INVALID_ID;
                    //columnTData[i].distributionNearestBOrders30s[j][rowIndex] = INVALID_ID;
                    columnCTData[i].distanceToNearestAOrderNavArea[j][rowIndex] = INVALID_ID;
                    columnCTData[i].distanceToNearestBOrderNavArea[j][rowIndex] = INVALID_ID;
                    columnCTData[i].distributionNearestAOrders[j][rowIndex] = INVALID_ID;
                    columnCTData[i].distributionNearestBOrders[j][rowIndex] = INVALID_ID;
                    //columnCTData[i].distributionNearestAOrders15s[j][rowIndex] = INVALID_ID;
                    //columnCTData[i].distributionNearestBOrders15s[j][rowIndex] = INVALID_ID;
                    //columnCTData[i].distributionNearestAOrders30s[j][rowIndex] = INVALID_ID;
                    //columnCTData[i].distributionNearestBOrders30s[j][rowIndex] = INVALID_ID;
                }
                for (int j = 0; j < num_places; j++) {
                    columnTData[i].curPlace[j][rowIndex] = false;
                    columnTData[i].distributionNearestPlace[j][rowIndex] = INVALID_ID;
                    //columnTData[i].distributionNearestPlace7to15s[j][rowIndex] = INVALID_ID;
                    columnCTData[i].curPlace[j][rowIndex] = false;
                    columnCTData[i].distributionNearestPlace[j][rowIndex] = INVALID_ID;
                    //columnCTData[i].distributionNearestPlace7to15s[j][rowIndex] = INVALID_ID;
                    for (int k = 0; k < num_prior_ticks; k++) {
                        columnTData[i].priorPlaces[k][j][rowIndex] = false;
                        columnCTData[i].priorPlaces[k][j][rowIndex] = false;
                    }
                }
                for (int j = 0; j < area_grid_size; j++) {
                    columnTData[i].areaGridCellInPlace[j][rowIndex] = false;
                    columnTData[i].distributionNearestAreaGridInPlace[j][rowIndex] = INVALID_ID;
                    //columnTData[i].distributionNearestAreaGridInPlace7to15s[j][rowIndex] = INVALID_ID;
                    columnCTData[i].areaGridCellInPlace[j][rowIndex] = false;
                    columnCTData[i].distributionNearestAreaGridInPlace[j][rowIndex] = INVALID_ID;
                    //columnCTData[i].distributionNearestAreaGridInPlace7to15s[j][rowIndex] = INVALID_ID;
                    for (int k = 0; k < num_prior_ticks; k++) {
                        columnTData[i].priorAreaGridCellInPlace[k][j][rowIndex] = false;
                        columnCTData[i].priorAreaGridCellInPlace[k][j][rowIndex] = false;
                    }
                }
                for (int j = 0; j < delta_pos_grid_num_cells; j++) {
                    columnTData[i].deltaPos[j][rowIndex] = false;
                    columnCTData[i].deltaPos[j][rowIndex] = false;
                }
            }
        }
    }

    size_t getAreaGridFlatIndex(Vec3 pos, AABB placeAABB) {
        double xPct = std::max(0., std::min(1., (pos.x - placeAABB.min.x) / (placeAABB.max.x - placeAABB.min.x)));
        double yPct = std::max(0., std::min(1., (pos.y - placeAABB.min.y) / (placeAABB.max.y - placeAABB.min.y)));
        size_t xValue = static_cast<size_t>(xPct * area_grid_dim);
        if (xValue == area_grid_dim) {
            xValue--;
        }
        size_t yValue = static_cast<size_t>(yPct * area_grid_dim);
        if (yValue == area_grid_dim) {
            yValue--;
        }
        return xValue + yValue * area_grid_dim;
    }

    void TeamFeatureStoreResult::commitTeamRow(FeatureStorePreCommitBuffer & buffer,
                                               DistanceToPlacesResult & distanceToPlaces,
                                               const nav_mesh::nav_file & navFile,
                                               int64_t roundIndex, int64_t tickIndex) {
        if (tickIndex % every_nth_row != 0) {
            return;
        }
        tickIndex /= every_nth_row;

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
        c4TicksSincePlant[tickIndex] = buffer.c4MapData.ticksSincePlant;
        int c4TimerBucket = std::min(num_c4_timer_buckets - 1,
                                     static_cast<int>(c4TicksSincePlant[tickIndex] / seconds_per_c4_timer_bucket));
        c4TimerBucketed[c4TimerBucket][tickIndex] = true;

        c4Pos[tickIndex] = buffer.c4MapData.c4Pos;
        c4DistanceToASite[tickIndex] =
            distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaId, a_site, navFile);
        c4DistanceToBSite[tickIndex] =
            distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaId, b_site, navFile);
        for (size_t j = 0; j < num_orders_per_site; j++) {
            float & aOrderDistance = c4DistanceToNearestAOrderNavArea[j][tickIndex];
            aOrderDistance = std::numeric_limits<double>::max();
            for (size_t k = 1; k < aOrders[j].places.size(); k++) {
                aOrderDistance = std::min(aOrderDistance,
                                          static_cast<float>(distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaIndex, aOrders[j].places[k])));
            }
            float & bOrderDistance = c4DistanceToNearestBOrderNavArea[j][tickIndex];
            bOrderDistance = std::numeric_limits<double>::max();
            for (size_t k = 1; k < bOrders[j].places.size(); k++) {
                bOrderDistance = std::min(bOrderDistance,
                                          static_cast<float>(distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaIndex, bOrders[j].places[k])));
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
            int64_t oldestHistoryIndex = buffer.getPlayerOldestContiguousHistoryIndex(btTeamPlayerData.playerId);

            /*
            if (columnIndex == 0 && btTeamPlayerData.teamId == ENGINE_TEAM_T) {
                t0Pos = btTeamPlayerData.curPos;
                areaIndex = btTeamPlayerData.curAreaIndex;
                areaId = btTeamPlayerData.curArea;
            }
             */

            if (columnIndex >= columnData.size()) {
                std::cout << "bad round index " << roundIndex << ", tickIndex " << tickIndex << std::endl;
            }
            columnData[columnIndex].playerId[tickIndex] = btTeamPlayerData.playerId;
            columnData[columnIndex].footPos[tickIndex] = btTeamPlayerData.curFootPos;
            columnData[columnIndex].velocity[tickIndex] = btTeamPlayerData.velocity;
            columnData[columnIndex].distanceToASite[tickIndex] =
                distanceToPlaces.getClosestDistance(btTeamPlayerData.curArea, a_site, navFile);
            columnData[columnIndex].distanceToBSite[tickIndex] =
                distanceToPlaces.getClosestDistance(btTeamPlayerData.curArea, b_site, navFile);
            for (size_t j = 0; j < num_orders_per_site; j++) {
                float & aOrderDistance = columnData[columnIndex].distanceToNearestAOrderNavArea[j][tickIndex];
                aOrderDistance = std::numeric_limits<double>::max();
                // start at 1 so skip tspawn (as all T's spend a significant time pre unfreeze in all orders then)
                for (size_t k = 1; k < aOrders[j].places.size(); k++) {
                    aOrderDistance = std::min(aOrderDistance,
                                              static_cast<float>(distanceToPlaces.getClosestDistance(btTeamPlayerData.curAreaIndex, aOrders[j].places[k])));
                }
                float & bOrderDistance = columnData[columnIndex].distanceToNearestBOrderNavArea[j][tickIndex];
                bOrderDistance = std::numeric_limits<double>::max();
                for (size_t k = 1; k < bOrders[j].places.size(); k++) {
                    bOrderDistance = std::min(bOrderDistance,
                                              static_cast<float>(distanceToPlaces.getClosestDistance(btTeamPlayerData.curAreaIndex, bOrders[j].places[k])));
                }
            }
            PlaceIndex curPlaceIndex = distanceToPlaces.getClosestValidPlace(btTeamPlayerData.curAreaIndex, navFile);
            string curPlaceString = navFile.get_place(curPlaceIndex);
            columnData[columnIndex].curPlace[curPlaceIndex][tickIndex] = true;
            size_t areaGridIndex = getAreaGridFlatIndex(btTeamPlayerData.curFootPos,
                                                        distanceToPlaces.placeToAABB.at(curPlaceString));
            columnData[columnIndex].areaGridCellInPlace[areaGridIndex][tickIndex] = true;
            for (int64_t j = 0; j < num_prior_ticks; j++) {
                int64_t priorTickIndex = (j + 1) * prior_tick_spacing;
                priorTickIndex = std::min(priorTickIndex, oldestHistoryIndex);
                const BTTeamPlayerData & priorBTTeamPlayerData =
                    buffer.historicalPlayerDataBuffer.fromNewest(priorTickIndex).at(btTeamPlayerData.playerId);
                columnData[columnIndex].priorFootPos[j][tickIndex] = priorBTTeamPlayerData.curFootPos;
                if (isnan(priorBTTeamPlayerData.curFootPos.x) || isnan(priorBTTeamPlayerData.curFootPos.y) || isnan(priorBTTeamPlayerData.curFootPos.z) ) {
                    std::cout << "found nan" << std::endl;
                }
                PlaceIndex priorPlaceIndex = distanceToPlaces.getClosestValidPlace(priorBTTeamPlayerData.curAreaIndex, navFile);
                string priorPlaceString = navFile.get_place(priorPlaceIndex);
                columnData[columnIndex].priorPlaces[j][priorPlaceIndex][tickIndex] = true;
                size_t priorAreaGridIndex = getAreaGridFlatIndex(priorBTTeamPlayerData.curFootPos,
                                                            distanceToPlaces.placeToAABB.at(priorPlaceString));
                columnData[columnIndex].priorAreaGridCellInPlace[j][priorAreaGridIndex][tickIndex] = true;
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

    void TeamFeatureStoreResult::computeOrderACausalLabels(int64_t curTick, CircularBuffer<int64_t> & futureTracker,
                                                           array<ColumnPlayerData, maxEnemies> & columnData,
                                                           ACausalTimingOption timingOption) {
        for (size_t playerColumn = 0; playerColumn < maxEnemies; playerColumn++) {
            /*
            if (curTick == 8240 && playerColumn == 4) {
                std::cout << "tick " << curTick << " player id " << columnData[playerColumn].playerId[curTick] << std::endl;
                for (size_t orderPerSite = 0; orderPerSite < num_orders_per_site; orderPerSite++) {
                    std::cout << "distribution nearest a orders 15s order " << orderPerSite << " " <<
                              columnData[playerColumn].distributionNearestAOrders15s[orderPerSite][curTick]
                              << std::endl;
                }
            }
             */
            if (columnData[playerColumn].playerId[curTick] == INVALID_ID) {
                continue;
            }
            // clear out values for current tick
            for (size_t orderPerSite = 0; orderPerSite < num_orders_per_site; orderPerSite++) {
                if (timingOption == ACausalTimingOption::s6) {
                    columnData[playerColumn].distributionNearestAOrders[orderPerSite][curTick] = 0;
                    columnData[playerColumn].distributionNearestBOrders[orderPerSite][curTick] = 0;
                }
                else if (timingOption == ACausalTimingOption::s15) {
                    //columnData[playerColumn].distributionNearestAOrders15s[orderPerSite][curTick] = 0;
                    //columnData[playerColumn].distributionNearestBOrders15s[orderPerSite][curTick] = 0;
                }
                else {
                    //columnData[playerColumn].distributionNearestAOrders30s[orderPerSite][curTick] = 0;
                    //columnData[playerColumn].distributionNearestBOrders30s[orderPerSite][curTick] = 0;
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
                        if (timingOption == ACausalTimingOption::s6) {
                            columnData[playerColumn].distributionNearestAOrders[orderPerSite][curTick]++;
                        }
                        else if (timingOption == ACausalTimingOption::s15) {
                            //columnData[playerColumn].distributionNearestAOrders15s[orderPerSite][curTick]++;
                        }
                        else {
                            //columnData[playerColumn].distributionNearestAOrders30s[orderPerSite][curTick]++;
                        }
                    }
                    for (const auto & orderPerSite : minDistanceBOrders) {
                        if (timingOption == ACausalTimingOption::s6) {
                            columnData[playerColumn].distributionNearestBOrders[orderPerSite][curTick]++;
                        }
                        else if (timingOption == ACausalTimingOption::s15) {
                            //columnData[playerColumn].distributionNearestBOrders15s[orderPerSite][curTick]++;
                        }
                        else {
                            //columnData[playerColumn].distributionNearestBOrders30s[orderPerSite][curTick]++;
                        }
                    }
                }
            }
            if (numPointsInDistribution != 0) {
                for (size_t orderPerSite = 0; orderPerSite < num_orders_per_site; orderPerSite++) {
                    if (timingOption == ACausalTimingOption::s6) {
                        columnData[playerColumn].distributionNearestAOrders[orderPerSite][curTick] /=
                            static_cast<double>(numPointsInDistribution);
                        columnData[playerColumn].distributionNearestBOrders[orderPerSite][curTick] /=
                            static_cast<double>(numPointsInDistribution);
                    }
                    else if (timingOption == ACausalTimingOption::s15) {
                        //columnData[playerColumn].distributionNearestAOrders15s[orderPerSite][curTick] /=
                        //    static_cast<double>(numPointsInDistribution);
                        //columnData[playerColumn].distributionNearestBOrders15s[orderPerSite][curTick] /=
                        //    static_cast<double>(numPointsInDistribution);
                    }
                    else {
                        //columnData[playerColumn].distributionNearestAOrders30s[orderPerSite][curTick] /=
                        //    static_cast<double>(numPointsInDistribution);
                        //columnData[playerColumn].distributionNearestBOrders30s[orderPerSite][curTick] /=
                        //    static_cast<double>(numPointsInDistribution);
                    }
                }
            }
        }
    }

    void TeamFeatureStoreResult::computePlaceACausalLabels(const Games & games, const Ticks & ticks, const TickRates & tickRates,
                                                           int64_t curGame, int64_t curTick, CircularBuffer<int64_t> &futureTracker,
                                                           array<ColumnPlayerData, maxEnemies> &columnData,
                                                           double futureSecondsThreshold,
                                                           const Players & players,
                                                           const DistanceToPlacesResult & distanceToPlacesResult,
                                                           const nav_mesh::nav_file & navFile) {
        for (size_t playerColumn = 0; playerColumn < maxEnemies; playerColumn++) {
            if (columnData[playerColumn].playerId[curTick] == INVALID_ID) {
                continue;
            }
            // clear out values for current tick
            for (size_t placeIndex = 0; placeIndex < num_places; placeIndex++) {
                columnData[playerColumn].distributionNearestPlace[placeIndex][curTick] = 0;
            }
            // want all points where alive, and accounting for ties
            size_t numPointsInDistribution = 0;
            for (int64_t futureTickIndex = 0; futureTickIndex < futureTracker.getCurSize(); futureTickIndex++) {
                int64_t futureTick = futureTracker.fromOldest(futureTickIndex);
                bool inWindow = secondsBetweenTicks(ticks, tickRates, curTick * every_nth_row, futureTick * every_nth_row) >=
                        futureSecondsThreshold;
                if (futureTick != curTick &&
                    columnData[playerColumn].playerId[curTick] == columnData[playerColumn].playerId[futureTick] &&
                    inWindow) {
                    numPointsInDistribution++;
                    for (size_t placeIndex = 0; placeIndex < num_places; placeIndex++) {
                        if (columnData[playerColumn].curPlace[placeIndex][futureTick]) {
                            columnData[playerColumn].distributionNearestPlace[placeIndex][curTick]++;
                        }
                    }
                }
                if (futureTick < curTick) {
                    std::cout << "ticks going wrong direction " << curTick << " " << futureTick << std::endl;
                    std::raise(SIGINT);
                }
            }
            if (numPointsInDistribution != 0) {
                for (size_t placeIndex = 0; placeIndex < num_places; placeIndex++) {
                    columnData[playerColumn].distributionNearestPlace[placeIndex][curTick] /=
                        static_cast<double>(numPointsInDistribution);
                }

                PlaceIndex curPlace = 50;
                int numCurPlaces = 0;
                for (size_t placeIndex = 0; placeIndex < num_places; placeIndex++) {
                    if (columnData[playerColumn].curPlace[placeIndex][curTick]) {
                        curPlace = placeIndex;
                        numCurPlaces++;
                    }
                }
                if (numCurPlaces != 1 || curPlace > 25) {
                    std::cout << "bad tick not one cur place " << curTick << std::endl;
                    std::raise(SIGINT);
                }
                (void) games;
                (void) curGame;
                (void) players;
                (void) distanceToPlacesResult;
                (void) navFile;
                /*
                int64_t curPlayerId = columnData[playerColumn].playerId[curTick];
                for (size_t nextPlaceIndex = 0; nextPlaceIndex < num_places; nextPlaceIndex++) {
                    if (columnData[playerColumn].distributionNearestPlace[nextPlaceIndex][curTick] > 0) {
                        double distanceBetweenCurAndNextPlace =
                                distanceToPlacesResult.getClosestDistance(distanceToPlacesResult.places[curPlace], distanceToPlacesResult.places[nextPlaceIndex], navFile);
                        if (secondsAwayAtMaxSpeed(distanceBetweenCurAndNextPlace) > 10.) {
                            std::cout << games.demoFile[curGame] << " bad tick reached (" << curPlace << ","
                                << distanceToPlacesResult.places[curPlace] << ") to (" << nextPlaceIndex << ","
                                << distanceToPlacesResult.places[nextPlaceIndex] << ") in under 2 seconds "
                                << " for player (" << curPlayerId << "," << players.name[curPlayerId + players.idOffset] << ") on tick id "
                                << curTick << " and game tick " << ticks.gameTickNumber[curTick] << " distance " << distanceBetweenCurAndNextPlace << std::endl;
                            for (int64_t futureTickIndex = 0; futureTickIndex < futureTracker.getCurSize(); futureTickIndex++) {
                                int64_t futureTick = futureTracker.fromOldest(futureTickIndex);
                                bool inWindow = secondsBetweenTicks(ticks, tickRates, curTick, futureTick) >=
                                                futureSecondsThreshold;
                                if (futureTick != curTick &&
                                    columnData[playerColumn].playerId[curTick] ==
                                    columnData[playerColumn].playerId[futureTick] &&
                                    inWindow && columnData[playerColumn].curPlace[nextPlaceIndex][futureTick]) {
                                    std::cout << "future tick " << futureTick << " and game tick " << ticks.gameTickNumber[futureTick] << std::endl;
                                }
                            }
                            std::raise(SIGINT);
                        }
                    }
                }
                 */
                int64_t curPlayerId = columnData[playerColumn].playerId[curTick];
                for (const PlaceIndex badCurPlace : {18, 3, 5, 20, 25/*17*//*11, 13, 17, 24*/}) {
                    for (const PlaceIndex badNextPlace : {11, 0/*11, 1, 2*//*7, 12*/})
                    if (curPlace == badCurPlace && columnData[playerColumn].distributionNearestPlace[badNextPlace][curTick] > 0) {
                        double distanceBetweenCurAndNextPlace =
                                distanceToPlacesResult.getClosestDistance(distanceToPlacesResult.places[curPlace], distanceToPlacesResult.places[badNextPlace], navFile);
                        std::cout << games.demoFile[curGame] << " bad tick reached (" << curPlace << ","
                                  << distanceToPlacesResult.places[curPlace] << ") to (" << badNextPlace << ","
                                  << distanceToPlacesResult.places[badNextPlace] << ") in under 2 seconds "
                                  << " for player (" << curPlayerId << "," << players.name[curPlayerId + players.idOffset] << ") on tick id "
                                  << curTick * every_nth_row << " and game tick " << ticks.gameTickNumber[curTick * every_nth_row] << " distance " << distanceBetweenCurAndNextPlace << std::endl;
                        for (int64_t futureTickIndex = 0; futureTickIndex < futureTracker.getCurSize(); futureTickIndex++) {
                            int64_t futureTick = futureTracker.fromOldest(futureTickIndex);
                            bool inWindow = secondsBetweenTicks(ticks, tickRates, curTick, futureTick) >=
                                            futureSecondsThreshold;
                            if (futureTick != curTick &&
                                columnData[playerColumn].playerId[curTick] ==
                                columnData[playerColumn].playerId[futureTick] &&
                                inWindow && columnData[playerColumn].curPlace[badNextPlace][futureTick]) {
                                std::cout << "future tick " << futureTick * every_nth_row << std::endl;
                            }
                        }
                        std::raise(SIGINT);
                    }
                }
            }
        }
    }

    void TeamFeatureStoreResult::computeAreaACausalLabels(const Ticks & ticks, const TickRates & tickRates,
                                                          int64_t curTick, CircularBuffer<int64_t> &futureTracker,
                                                          array<ColumnPlayerData, maxEnemies> &columnData,
                                                          double futureSecondsTheshold) {
        for (size_t playerColumn = 0; playerColumn < maxEnemies; playerColumn++) {
            if (columnData[playerColumn].playerId[curTick] == INVALID_ID) {
                continue;
            }
            // clear out values for current tick
            for (size_t areaGridIndex = 0; areaGridIndex < area_grid_size; areaGridIndex++) {
                columnData[playerColumn].distributionNearestAreaGridInPlace[areaGridIndex][curTick] = 0;
                //columnData[playerColumn].distributionNearestAreaGridInPlace7to15s[areaGridIndex][curTick] = 0;
            }
            // want all points where alive, and accounting for ties
            size_t numPointsInDistribution = 0;
            for (int64_t futureTickIndex = 0; futureTickIndex < futureTracker.getCurSize(); futureTickIndex++) {
                int64_t futureTick = futureTracker.fromOldest(futureTickIndex);
                bool inWindow = secondsBetweenTicks(ticks, tickRates, curTick * every_nth_row, futureTick * every_nth_row) >=
                        futureSecondsTheshold;
                if (futureTick != curTick &&
                    columnData[playerColumn].playerId[curTick] == columnData[playerColumn].playerId[futureTick] &&
                    inWindow) {
                    numPointsInDistribution++;
                    for (size_t areaGridIndex = 0; areaGridIndex < area_grid_size; areaGridIndex++) {
                        if (columnData[playerColumn].areaGridCellInPlace[areaGridIndex][futureTick]) {
                            columnData[playerColumn].distributionNearestAreaGridInPlace[areaGridIndex][curTick]++;
                        }
                    }
                }
            }
            if (numPointsInDistribution != 0) {
                for (size_t areaGridIndex = 0; areaGridIndex < area_grid_size; areaGridIndex++) {
                    columnData[playerColumn].distributionNearestAreaGridInPlace[areaGridIndex][curTick] /=
                            static_cast<double>(numPointsInDistribution);
                    //columnData[playerColumn].distributionNearestAreaGridInPlace7to15s[areaGridIndex][curTick] /=
                    //    static_cast<double>(numPointsInDistribution);
                }
            }
        }
    }

    void TeamFeatureStoreResult::computeDeltaPosACausalLabels(int64_t curTick, CircularBuffer<int64_t> & futureTracker,
                                                              array<ColumnPlayerData,maxEnemies> & columnData) {
        for (size_t playerColumn = 0; playerColumn < maxEnemies; playerColumn++) {
            if (columnData[playerColumn].playerId[curTick] == INVALID_ID) {
                continue;
            }
            // clear out values for current tick
            for (size_t deltaPosGridIndex = 0; deltaPosGridIndex < delta_pos_grid_num_cells; deltaPosGridIndex++) {
                columnData[playerColumn].deltaPos[deltaPosGridIndex][curTick] = false;
            }
            if (futureTracker.isEmpty()) {
                std::cout << "delta pos acausal label with no future ticks" << std::endl;
                std::raise(SIGINT);
            }
            int64_t futureTickIndex = futureTracker.fromOldest();
            if (futureTickIndex < curTick) {
                std::cout << "delta pos acausal future tick in past" << std::endl;
                std::raise(SIGINT);
            }
            Vec3 curFootPos = columnData[playerColumn].footPos[curTick];
            AABB deltaPosRange = {
                    {
                        curFootPos.x - delta_pos_grid_radius,
                        curFootPos.y - delta_pos_grid_radius,
                        curFootPos.z
                    },
                    {
                        curFootPos.x + delta_pos_grid_radius,
                        curFootPos.y + delta_pos_grid_radius,
                        curFootPos.z
                    }
            };
            int deltaPosIndex = getAreaGridFlatIndex(columnData[playerColumn].footPos[futureTickIndex], deltaPosRange);
            columnData[playerColumn].deltaPos[deltaPosIndex][curTick] = true;
        }
    }

    void TeamFeatureStoreResult::computeAcausalLabels(const Games & games, const Rounds & rounds,
                                                      const Ticks & ticks,
                                                      const Players & players,
                                                      const DistanceToPlacesResult & distanceToPlacesResult,
                                                      const nav_mesh::nav_file & navFile,
                                                      const csknow::key_retake_events::KeyRetakeEvents & keyRetakeEvents) {
        std::atomic<int64_t> roundsProcessed = 0;
        /*
        for (size_t i = 0; i < columnTData[4].distributionNearestAOrders15s[0].size(); i++) {
            if (std::isnan(columnTData[4].distributionNearestAOrders15s[0][i])) {
                std::cout << i << " is nan with tick index " << std::endl;
            }
        }
         */
//#pragma omp parallel for
        for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
            int64_t gameIndex = rounds.gameId[roundIndex];
            TickRates tickRates = computeTickRates(games, rounds, roundIndex);
            CircularBuffer<int64_t> ticks1sFutureTracker(4), ticks2sFutureTracker(4), ticks6sFutureTracker(6),
                // this one I add to every frame and remove when too far in the future, more accuracte
                bothSidesTicks1sFutureTracker(10);
            //, ticks15sFutureTracker(15), ticks30sFutureTracker(30);
            for (int64_t unmodifiedTickIndex = rounds.ticksPerRound[roundIndex].maxId;
                 unmodifiedTickIndex >= rounds.ticksPerRound[roundIndex].minId; unmodifiedTickIndex--) {
                if (unmodifiedTickIndex % every_nth_row != 0) {
                    continue;
                }
                int64_t tickIndex = unmodifiedTickIndex / every_nth_row;
                gameTickNumber[tickIndex] = ticks.gameTickNumber[unmodifiedTickIndex];
                roundNumber[tickIndex] = rounds.roundNumber[roundIndex];
                freezeTimeEnded[tickIndex] = rounds.freezeTimeEnd[roundIndex] <= unmodifiedTickIndex;
                retakeSaveRoundTick[tickIndex] = keyRetakeEvents.ctAliveAfterExplosion[unmodifiedTickIndex] ||
                        keyRetakeEvents.ctAliveAfterExplosion[unmodifiedTickIndex];
                // add a new tick every second
                if (ticks1sFutureTracker.isEmpty() ||
                    secondsBetweenTicks(ticks, tickRates, tickIndex * every_nth_row, ticks1sFutureTracker.fromNewest() * every_nth_row) >= 0.25) {
                    ticks1sFutureTracker.enqueue(tickIndex);
                }
                if (ticks2sFutureTracker.isEmpty() ||
                    secondsBetweenTicks(ticks, tickRates, tickIndex * every_nth_row, ticks2sFutureTracker.fromNewest() * every_nth_row) >= 0.5) {
                    ticks2sFutureTracker.enqueue(tickIndex);
                }
                if (ticks6sFutureTracker.isEmpty() ||
                    secondsBetweenTicks(ticks, tickRates, tickIndex * every_nth_row, ticks6sFutureTracker.fromNewest() * every_nth_row) >= 1.) {
                    ticks6sFutureTracker.enqueue(tickIndex);
                }
                bothSidesTicks1sFutureTracker.enqueue(tickIndex);
                while (secondsBetweenTicks(ticks, tickRates, tickIndex * every_nth_row, bothSidesTicks1sFutureTracker.fromOldest() * every_nth_row) > 1.) {
                    bothSidesTicks1sFutureTracker.dequeue();
                }
                /*
                if (ticks15sFutureTracker.isEmpty() ||
                    secondsBetweenTicks(ticks, tickRates, tickIndex, ticks15sFutureTracker.fromNewest()) >= 1.) {
                    ticks15sFutureTracker.enqueue(tickIndex);
                }
                if (ticks30sFutureTracker.isEmpty() ||
                    secondsBetweenTicks(ticks, tickRates, tickIndex, ticks30sFutureTracker.fromNewest()) >= 1.) {
                    ticks30sFutureTracker.enqueue(tickIndex);
                }
                 */
                if (ticks.roundId[ticks1sFutureTracker.fromOldest() * every_nth_row] != ticks.roundId[tickIndex * every_nth_row]) {
                    std::cout << "round id mismatch cur tick " << tickIndex * every_nth_row << " future tick " << ticks1sFutureTracker.fromOldest() * every_nth_row << std::endl;
                    std::raise(SIGINT);
                }
                computeOrderACausalLabels(tickIndex, ticks6sFutureTracker, columnCTData, ACausalTimingOption::s6);
                //computeOrderACausalLabels(tickIndex, ticks15sFutureTracker, columnCTData, ACausalTimingOption::s15);
                //computeOrderACausalLabels(tickIndex, ticks30sFutureTracker, columnCTData, ACausalTimingOption::s30);
                computeOrderACausalLabels(tickIndex, ticks6sFutureTracker, columnTData, ACausalTimingOption::s6);
                //computeOrderACausalLabels(tickIndex, ticks15sFutureTracker, columnTData, ACausalTimingOption::s15);
                //computeOrderACausalLabels(tickIndex, ticks30sFutureTracker, columnTData, ACausalTimingOption::s30);
                computePlaceACausalLabels(games, ticks, tickRates, gameIndex, tickIndex, ticks2sFutureTracker, columnCTData, 0.1,
                                          players, distanceToPlacesResult, navFile);
                computePlaceACausalLabels(games, ticks, tickRates, gameIndex, tickIndex, ticks2sFutureTracker, columnTData, 0.1,
                                          players, distanceToPlacesResult, navFile);
                computeAreaACausalLabels(ticks, tickRates, tickIndex, ticks1sFutureTracker, columnCTData, 0.1);
                computeAreaACausalLabels(ticks, tickRates, tickIndex, ticks1sFutureTracker, columnTData, 0.1);
                computeDeltaPosACausalLabels(tickIndex, bothSidesTicks1sFutureTracker, columnCTData);
                computeDeltaPosACausalLabels(tickIndex, bothSidesTicks1sFutureTracker, columnTData);
                //computePlaceAreaACausalLabels(ticks, tickRates, tickIndex, ticks15sFutureTracker, columnCTData);
                //computePlaceAreaACausalLabels(ticks, tickRates, tickIndex, ticks15sFutureTracker, columnTData);
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
                /*
                for (size_t columnIndex = 0; columnIndex < maxEnemies; columnIndex++) {
                    if (columnTData[columnIndex].playerId[tickIndex] != INVALID_ID &&
                        columnTData[columnIndex].curPlace[7][tickIndex] && !retakeSaveRoundTick[tickIndex] &&
                        (c4Status[tickIndex] == C4Status::PlantedB || c4Status[tickIndex] == C4Status::PlantedA)) {
                        int64_t badPlayerIndex = columnTData[columnIndex].playerId[tickIndex];
                        std::cout << games.demoFile[gameIndex] << " bad tick "
                                  << " for player (" << badPlayerIndex << "," << players.name[badPlayerIndex + players.idOffset] << ") on tick id "
                                  << unmodifiedTickIndex << " and game tick " << ticks.gameTickNumber[unmodifiedTickIndex] << std::endl;
                        std::raise(SIGINT);
                    }
                }
                 */
                if (ticks.gameTickNumber[unmodifiedTickIndex] == 236466) {
                    std::cout << "good round " << roundIndex << std::endl;
                    std::raise(SIGINT);
                }
            }
            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
        }
    }

    void TeamFeatureStoreResult::toHDF5Inner(HighFive::File & file) {
        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        //hdf5FlatCreateProps.add(HighFive::Deflate(6));
        //hdf5FlatCreateProps.add(HighFive::Chunking(roundId.size()));

        file.createDataSet("/data/round id", roundId, hdf5FlatCreateProps);
        file.createDataSet("/data/tick id", tickId, hdf5FlatCreateProps);
        file.createDataSet("/data/game tick number", gameTickNumber, hdf5FlatCreateProps);
        file.createDataSet("/data/round number", roundNumber, hdf5FlatCreateProps);
        file.createDataSet("/data/valid", valid, hdf5FlatCreateProps);
        file.createDataSet("/data/freeze time ended", freezeTimeEnded, hdf5FlatCreateProps);
        file.createDataSet("/data/retake save round tick", retakeSaveRoundTick, hdf5FlatCreateProps);
        file.createDataSet("/data/c4 status", vectorOfEnumsToVectorOfInts(c4Status), hdf5FlatCreateProps);
        file.createDataSet("/data/c4 ticks since plant", c4TicksSincePlant, hdf5FlatCreateProps);
        for (size_t c4TimerBucketIndex = 0; c4TimerBucketIndex < num_c4_timer_buckets; c4TimerBucketIndex++) {
            file.createDataSet("/data/c4 timer bucketed " + std::to_string(c4TimerBucketIndex), c4TimerBucketed[c4TimerBucketIndex], hdf5FlatCreateProps);
        }
        saveVec3VectorToHDF5(c4Pos, file, "c4 pos", hdf5FlatCreateProps);
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
                for (int indexOnTeam = 0; indexOnTeam < maxEnemies; indexOnTeam++) {
                    file.createDataSet("/data/player index on team " + std::to_string(indexOnTeam) + " " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].indexOnTeam[indexOnTeam], hdf5FlatCreateProps);
                }
                file.createDataSet("/data/player ctTeam " + columnTeam + " " + iStr,
                                   columnData[columnPlayer].ctTeam, hdf5FlatCreateProps);
                saveVec3VectorToHDF5(columnData[columnPlayer].footPos, file,
                                     "player pos " + columnTeam + " " + iStr, hdf5FlatCreateProps);
                for (int priorTick = 0; priorTick < num_prior_ticks; priorTick++) {
                    saveVec3VectorToHDF5(columnData[columnPlayer].priorFootPos[priorTick], file,
                                         "player pos " + columnTeam + " " + iStr + " t-" + std::to_string(priorTick+1),
                                         hdf5FlatCreateProps);
                }
                saveVec3VectorToHDF5(columnData[columnPlayer].velocity, file,
                                     "player velocity " + columnTeam + " " + iStr, hdf5FlatCreateProps);
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
                    file.createDataSet("/data/distribution nearest a order " + orderIndexStr + " " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].distributionNearestAOrders[orderIndex], hdf5FlatCreateProps);
                    file.createDataSet("/data/distribution nearest b order " + orderIndexStr + " " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].distributionNearestBOrders[orderIndex], hdf5FlatCreateProps);
                    //file.createDataSet("/data/distribution nearest a order " + orderIndexStr + " 15s " + columnTeam + " " + iStr,
                    //                   columnData[columnPlayer].distributionNearestAOrders15s[orderIndex], hdf5FlatCreateProps);
                    //file.createDataSet("/data/distribution nearest b order " + orderIndexStr + " 15s " + columnTeam + " " + iStr,
                    //                   columnData[columnPlayer].distributionNearestBOrders15s[orderIndex], hdf5FlatCreateProps);
                    //file.createDataSet("/data/distribution nearest a order " + orderIndexStr + " 30s " + columnTeam + " " + iStr,
                    //                   columnData[columnPlayer].distributionNearestAOrders30s[orderIndex], hdf5FlatCreateProps);
                    //file.createDataSet("/data/distribution nearest b order " + orderIndexStr + " 30s " + columnTeam + " " + iStr,
                    //                   columnData[columnPlayer].distributionNearestBOrders30s[orderIndex], hdf5FlatCreateProps);
                }
                for (size_t placeIndex = 0; placeIndex < num_places; placeIndex++) {
                    string placeIndexStr = std::to_string(placeIndex);
                    file.createDataSet("/data/cur place " + placeIndexStr + " " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].curPlace[placeIndex], hdf5FlatCreateProps);
                    file.createDataSet("/data/distribution nearest place " + placeIndexStr + " " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].distributionNearestPlace[placeIndex], hdf5FlatCreateProps);
                    //file.createDataSet("/data/distribution nearest place 7 to 15s " + placeIndexStr + " " + columnTeam + " " + iStr,
                    //                   columnData[columnPlayer].distributionNearestPlace7to15s[placeIndex], hdf5FlatCreateProps);
                    /*
                    for (int priorTick = 0; priorTick < num_prior_ticks; priorTick++) {
                        file.createDataSet("/data/prior place " + placeIndexStr + " " + columnTeam + " " + iStr +  " t-" + std::to_string(priorTick+1),
                                           columnData[columnPlayer].priorPlaces[priorTick][placeIndex], hdf5FlatCreateProps);
                    }
                     */
                }
                for (size_t areaGridIndex = 0; areaGridIndex < area_grid_size; areaGridIndex++) {
                    string areaGridIndexStr = std::to_string(areaGridIndex);
                    file.createDataSet("/data/area grid cell in place " + areaGridIndexStr + " " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].areaGridCellInPlace[areaGridIndex], hdf5FlatCreateProps);
                    file.createDataSet("/data/distribution nearest area grid in place " + areaGridIndexStr + " " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].distributionNearestAreaGridInPlace[areaGridIndex], hdf5FlatCreateProps);
                    //file.createDataSet("/data/distribution nearest area grid in place 7 to 15s " + areaGridIndexStr + " " + columnTeam + " " + iStr,
                    //                   columnData[columnPlayer].distributionNearestAreaGridInPlace7to15s[areaGridIndex], hdf5FlatCreateProps);
                    /*
                    for (int priorTick = 0; priorTick < num_prior_ticks; priorTick++) {
                        file.createDataSet("/data/prior area grid cell in place " + areaGridIndexStr + " " + columnTeam + " " + iStr +  " t-" + std::to_string(priorTick+1),
                                           columnData[columnPlayer].priorAreaGridCellInPlace[priorTick][areaGridIndex], hdf5FlatCreateProps);
                    }
                     */
                }
                for (size_t deltaPosIndex = 0; deltaPosIndex < delta_pos_grid_num_cells; deltaPosIndex++) {
                    string deltaPosIndexStr = std::to_string(deltaPosIndex);
                    file.createDataSet("/data/delta pos " + deltaPosIndexStr + " " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].deltaPos[deltaPosIndex], hdf5FlatCreateProps);
                }
            }
        }
    }
}
