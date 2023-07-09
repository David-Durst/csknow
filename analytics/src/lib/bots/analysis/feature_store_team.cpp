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
    void TeamFeatureStoreResult::init(size_t size) {
        gameId.resize(size, INVALID_ID);
        roundId.resize(size, INVALID_ID);
        roundNumber.resize(size, INVALID_ID);
        tickId.resize(size, INVALID_ID);
        gameTickNumber.resize(size, INVALID_ID);
        valid.resize(size, false);
        freezeTimeEnded.resize(size, false);
        retakeSaveRoundTick.resize(size, false);
        testName.resize(size, "INVALID");
        testSuccess.resize(size, false);
        baiting.resize(size, false);
        c4Status.resize(size, C4Status::NotPlanted);
        c4PlantA.resize(size, false);
        c4PlantB.resize(size, false);
        c4NotPlanted.resize(size, false);
        c4TicksSincePlant.resize(size, INVALID_ID);
        for (int j = 0; j < num_c4_timer_buckets; j++) {
            c4TimerBucketed[j].resize(size, false);
        }
        c4Pos.resize(size, zeroVec);
        c4DistanceToASite.resize(size, INVALID_ID);
        c4DistanceToBSite.resize(size, INVALID_ID);
        /*
        for (int j = 0; j < num_orders_per_site; j++) {
            c4DistanceToNearestAOrderNavArea[j].resize(size, INVALID_ID);
            c4DistanceToNearestBOrderNavArea[j].resize(size, INVALID_ID);
        }
         */
        for (int i = 0; i < maxEnemies; i++) {
            for (int j = 0; j < maxEnemies; j++) {
                columnTData[i].indexOnTeam[j].resize(size, j == i);
                columnCTData[i].indexOnTeam[j].resize(size, j == i);
            }
            columnTData[i].playerId.resize(size, INVALID_ID);
            columnTData[i].ctTeam.resize(size, false);
            columnTData[i].alive.resize(size, false);
            columnTData[i].footPos.resize(size, zeroVec);
            //columnTData[i].alignedFootPos.resize(size, zeroVec);
            columnTData[i].velocity.resize(size, zeroVec);
            //columnTData[i].distanceToASite.resize(size, 0);
            //columnTData[i].distanceToBSite.resize(size, 0);
            columnCTData[i].playerId.resize(size, INVALID_ID);
            columnCTData[i].ctTeam.resize(size, true);
            columnCTData[i].alive.resize(size, false);
            columnCTData[i].footPos.resize(size, zeroVec);
            //columnCTData[i].alignedFootPos.resize(size, zeroVec);
            columnCTData[i].velocity.resize(size, zeroVec);
            //columnCTData[i].distanceToASite.resize(size, 0);
            //columnCTData[i].distanceToBSite.resize(size, 0);
            for (int j = 0; j < num_prior_ticks; j++) {
                columnTData[i].priorFootPos[j].resize(size, zeroVec);
                columnTData[i].priorVelocity[j].resize(size, zeroVec);
                columnTData[i].priorFootPosValid[j].resize(size, false);
                columnCTData[i].priorFootPos[j].resize(size, zeroVec);
                columnCTData[i].priorVelocity[j].resize(size, zeroVec);
                columnCTData[i].priorFootPosValid[j].resize(size, false);
            }
            /*
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
            }
            for (int j = 0; j < area_grid_size; j++) {
                columnTData[i].areaGridCellInPlace[j].resize(size, false);
                columnTData[i].distributionNearestAreaGridInPlace[j].resize(size, INVALID_ID);
                //columnTData[i].distributionNearestAreaGridInPlace7to15s[j].resize(size, INVALID_ID);
                columnCTData[i].areaGridCellInPlace[j].resize(size, false);
                columnCTData[i].distributionNearestAreaGridInPlace[j].resize(size, INVALID_ID);
                //columnCTData[i].distributionNearestAreaGridInPlace7to15s[j].resize(size, INVALID_ID);
            }
             */
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

    TeamFeatureStoreResult::TeamFeatureStoreResult() : TeamFeatureStoreResult(1, {}) { }

    TeamFeatureStoreResult::TeamFeatureStoreResult(size_t size, const std::vector<csknow::orders::QueryOrder> & orders,
                                                   std::optional<std::reference_wrapper<const Ticks>> ticks,
                                                   std::optional<std::reference_wrapper<const csknow::key_retake_events::KeyRetakeEvents>> keyRetakeEvents) {
        tickIdToInternalId.resize(size, INVALID_ID);
        size_t internalSize = 0;
        if (keyRetakeEvents && ticks) {
            int64_t nextTickId = 0;
            const Ticks & refTicks = ticks->get();
            const csknow::key_retake_events::KeyRetakeEvents & refKeyRetakeEvents = keyRetakeEvents->get();
            for (int64_t i = 0; i < static_cast<int64_t>(size); i++) {
                int64_t roundIndex = refTicks.roundId[i];
                if (roundIndex == INVALID_ID) {
                    continue;
                }
                bool testCondition = (refKeyRetakeEvents.roundHasCompleteTest[roundIndex] || refKeyRetakeEvents.roundHasFailedTest[roundIndex]) &&
                        (refKeyRetakeEvents.tAlive[i] || refKeyRetakeEvents.ctAlive[i]) &&
                        refKeyRetakeEvents.testStartBeforeOrDuringThisTick[i] && !refKeyRetakeEvents.testEndBeforeOrDuringThisTick[i];
                bool nonTestCondition = refKeyRetakeEvents.enableNonTestPlantRounds && refKeyRetakeEvents.roundHasPlant[roundIndex] &&
                        refKeyRetakeEvents.plantFinishedBeforeOrDuringThisTick[i] && refKeyRetakeEvents.ctAlive[i] && refKeyRetakeEvents.tAlive[i] &&
                        !(refKeyRetakeEvents.explosionBeforeOrDuringThisTick[i] || refKeyRetakeEvents.defusalFinishedBeforeOrDuringThisTick[i]);
                if (testCondition || nonTestCondition) {
                    if (i % every_nth_row != 0) {
                        continue;
                    }
                    tickIdToInternalId[i] = nextTickId;
                    nextTickId++;
                }
            }
            internalSize = static_cast<size_t>(nextTickId);
        }
        else {
            for (int64_t i = 0; i < static_cast<int64_t>(size); i++) {
                tickIdToInternalId[i] = i;
            }
            internalSize = size;
        }

        internalIdToTickId.resize(internalSize, INVALID_ID);
        for (size_t i = 0; i < size; i++) {
            if (tickIdToInternalId[i] != INVALID_ID) {
                internalIdToTickId[tickIdToInternalId[i]] = static_cast<int64_t>(i);
            }
        }

        init(internalSize);
        setOrders(orders);
    }
    
    void TeamFeatureStoreResult::reinit() {
        for (int64_t rowIndex = 0; rowIndex < size; rowIndex++) {
            gameId[rowIndex] = INVALID_ID;
            roundId[rowIndex] = INVALID_ID;
            roundNumber[rowIndex] = INVALID_ID;
            tickId[rowIndex] = INVALID_ID;
            gameTickNumber[rowIndex] = INVALID_ID;
            valid[rowIndex] = false;
            freezeTimeEnded[rowIndex] = false;
            retakeSaveRoundTick[rowIndex] = false;
            testName[rowIndex] = "INVALID";
            testSuccess[rowIndex] = false;
            c4Status[rowIndex] = C4Status::NotPlanted;
            c4PlantA[rowIndex] = false;
            c4PlantB[rowIndex] = false;
            c4NotPlanted[rowIndex] = false;
            c4TicksSincePlant[rowIndex] = INVALID_ID;
            for (int j = 0; j < num_c4_timer_buckets; j++) {
                c4TimerBucketed[j][rowIndex] = INVALID_ID;
            }
            c4Pos[rowIndex] = zeroVec;
            c4DistanceToASite[rowIndex] = INVALID_ID;
            c4DistanceToBSite[rowIndex] = INVALID_ID;
            /*
            for (int j = 0; j < num_orders_per_site; j++) {
                c4DistanceToNearestAOrderNavArea[j][rowIndex] = INVALID_ID;
                c4DistanceToNearestBOrderNavArea[j][rowIndex] = INVALID_ID;
            }
             */
            for (int i = 0; i < maxEnemies; i++) {
                for (int j = 0; j < maxEnemies; j++) {
                    columnTData[i].indexOnTeam[j][rowIndex] = j == i;
                    columnCTData[i].indexOnTeam[j][rowIndex] = j == i;
                }
                columnTData[i].playerId[rowIndex] = INVALID_ID;
                columnTData[i].ctTeam[rowIndex] = false;
                columnTData[i].alive[rowIndex] = false;
                columnTData[i].footPos[rowIndex] = zeroVec;
                //columnTData[i].alignedFootPos[rowIndex] = zeroVec;
                columnTData[i].velocity[rowIndex] = zeroVec;
                //columnTData[i].distanceToASite[rowIndex] = 0.;
                //columnTData[i].distanceToBSite[rowIndex] = 0.;
                columnCTData[i].playerId[rowIndex] = INVALID_ID;
                columnCTData[i].ctTeam[rowIndex] = true;
                columnCTData[i].alive[rowIndex] = false;
                columnCTData[i].footPos[rowIndex] = zeroVec;
                //columnCTData[i].alignedFootPos[rowIndex] = zeroVec;
                columnCTData[i].velocity[rowIndex] = zeroVec;
                //columnCTData[i].distanceToASite[rowIndex] = 0.;
                //columnCTData[i].distanceToBSite[rowIndex] = 0.;
                for (int j = 0; j < num_prior_ticks; j++) {
                    columnTData[i].priorFootPos[j][rowIndex] = zeroVec;
                    columnTData[i].priorVelocity[j][rowIndex] = zeroVec;
                    columnTData[i].priorFootPosValid[j][rowIndex] = false;
                    columnCTData[i].priorFootPos[j][rowIndex] = zeroVec;
                    columnCTData[i].priorVelocity[j][rowIndex] = zeroVec;
                    columnCTData[i].priorFootPosValid[j][rowIndex] = false;
                }
                /*
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
                }
                for (int j = 0; j < area_grid_size; j++) {
                    columnTData[i].areaGridCellInPlace[j][rowIndex] = false;
                    columnTData[i].distributionNearestAreaGridInPlace[j][rowIndex] = INVALID_ID;
                    //columnTData[i].distributionNearestAreaGridInPlace7to15s[j][rowIndex] = INVALID_ID;
                    columnCTData[i].areaGridCellInPlace[j][rowIndex] = false;
                    columnCTData[i].distributionNearestAreaGridInPlace[j][rowIndex] = INVALID_ID;
                    //columnCTData[i].distributionNearestAreaGridInPlace7to15s[j][rowIndex] = INVALID_ID;
                }
                 */
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
        int64_t internalTickIndex = tickIdToInternalId[tickIndex];
        if (internalTickIndex == INVALID_ID) {
            return;
        }

        roundId[internalTickIndex] = roundIndex;
        tickId[internalTickIndex] = internalTickIndex;
        valid[internalTickIndex] = !buffer.btTeamPlayerData.empty();

        if (buffer.c4MapData.c4Planted) {
            double c4DistanceToASite =
                distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaId, a_site, navFile);
            double c4DistanceToBSite =
                distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaId, b_site, navFile);
            c4Status[internalTickIndex] = c4DistanceToASite < c4DistanceToBSite ? C4Status::PlantedA : C4Status::PlantedB;
            if (c4Status[internalTickIndex] == C4Status::PlantedA) {
                c4PlantA[internalTickIndex] = true;
            }
            else {
                c4PlantB[internalTickIndex] = true;
            }
        }
        else {
            c4Status[internalTickIndex] = C4Status::NotPlanted;
            c4NotPlanted[internalTickIndex] = true;
        }
        c4TicksSincePlant[internalTickIndex] = buffer.c4MapData.ticksSincePlant;
        int c4TimerBucket = std::min(num_c4_timer_buckets - 1,
                                     static_cast<int>(c4TicksSincePlant[internalTickIndex] / seconds_per_c4_timer_bucket));
        c4TimerBucketed[c4TimerBucket][internalTickIndex] = true;

        c4Pos[internalTickIndex] = buffer.c4MapData.c4Pos;
        c4DistanceToASite[internalTickIndex] =
            distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaId, a_site, navFile);
        c4DistanceToBSite[internalTickIndex] =
            distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaId, b_site, navFile);
        /*
        for (size_t j = 0; j < num_orders_per_site; j++) {
            float & aOrderDistance = c4DistanceToNearestAOrderNavArea[j][internalTickIndex];
            aOrderDistance = std::numeric_limits<double>::max();
            for (size_t k = 1; k < aOrders[j].places.size(); k++) {
                aOrderDistance = std::min(aOrderDistance,
                                          static_cast<float>(distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaIndex, aOrders[j].places[k])));
            }
            float & bOrderDistance = c4DistanceToNearestBOrderNavArea[j][internalTickIndex];
            bOrderDistance = std::numeric_limits<double>::max();
            for (size_t k = 1; k < bOrders[j].places.size(); k++) {
                bOrderDistance = std::min(bOrderDistance,
                                          static_cast<float>(distanceToPlaces.getClosestDistance(buffer.c4MapData.c4AreaIndex, bOrders[j].places[k])));
            }
        }
         */

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
                std::cout << "bad round index " << roundIndex << ", internalTickIndex " << internalTickIndex << std::endl;
            }
            columnData[columnIndex].playerId[internalTickIndex] = btTeamPlayerData.playerId;
            columnData[columnIndex].alive[internalTickIndex] = true;
            columnData[columnIndex].footPos[internalTickIndex] = btTeamPlayerData.curFootPos;
            //columnData[columnIndex].alignedFootPos[internalTickIndex] = (btTeamPlayerData.curFootPos / delta_pos_grid_num_cells_per_xy_dim).trunc();
            columnData[columnIndex].velocity[internalTickIndex] = btTeamPlayerData.velocity;
            //columnData[columnIndex].distanceToASite[internalTickIndex] =
            //    distanceToPlaces.getClosestDistance(btTeamPlayerData.curArea, a_site, navFile);
            //columnData[columnIndex].distanceToBSite[internalTickIndex] =
            //    distanceToPlaces.getClosestDistance(btTeamPlayerData.curArea, b_site, navFile);
            /*
            for (size_t j = 0; j < num_orders_per_site; j++) {
                float & aOrderDistance = columnData[columnIndex].distanceToNearestAOrderNavArea[j][internalTickIndex];
                aOrderDistance = std::numeric_limits<double>::max();
                // start at 1 so skip tspawn (as all T's spend a significant time pre unfreeze in all orders then)
                for (size_t k = 1; k < aOrders[j].places.size(); k++) {
                    aOrderDistance = std::min(aOrderDistance,
                                              static_cast<float>(distanceToPlaces.getClosestDistance(btTeamPlayerData.curAreaIndex, aOrders[j].places[k])));
                }
                float & bOrderDistance = columnData[columnIndex].distanceToNearestBOrderNavArea[j][internalTickIndex];
                bOrderDistance = std::numeric_limits<double>::max();
                for (size_t k = 1; k < bOrders[j].places.size(); k++) {
                    bOrderDistance = std::min(bOrderDistance,
                                              static_cast<float>(distanceToPlaces.getClosestDistance(btTeamPlayerData.curAreaIndex, bOrders[j].places[k])));
                }
            }
            PlaceIndex curPlaceIndex = distanceToPlaces.getClosestValidPlace(btTeamPlayerData.curAreaIndex, navFile);
            string curPlaceString = navFile.get_place(curPlaceIndex);
            columnData[columnIndex].curPlace[curPlaceIndex][internalTickIndex] = true;
            size_t areaGridIndex = getAreaGridFlatIndex(btTeamPlayerData.curFootPos,
                                                        distanceToPlaces.placeToAABB.at(curPlaceString));
            columnData[columnIndex].areaGridCellInPlace[areaGridIndex][internalTickIndex] = true;
             */
            for (int64_t j = 0; j < num_prior_ticks; j++) {
                int64_t priorTickIndex = (j + 1) * prior_tick_spacing;
                priorTickIndex = std::min(priorTickIndex, oldestHistoryIndex);
                bool setPlayerPriorFootPos = false;
                const map<CSGOId, BTTeamPlayerData> & priorBTTeamTickData =
                        buffer.historicalPlayerDataBuffer.fromNewest(priorTickIndex);
                if (buffer.historicalPlayerDataBuffer.getCurSize() > priorTickIndex) {
                    if (priorBTTeamTickData.count(btTeamPlayerData.playerId) > 0) {
                        setPlayerPriorFootPos = true;
                        const BTTeamPlayerData & priorBTTeamPlayerData =
                                priorBTTeamTickData.at(btTeamPlayerData.playerId);
                        columnData[columnIndex].priorFootPos[j][internalTickIndex] = priorBTTeamPlayerData.curFootPos;
                        columnData[columnIndex].priorVelocity[j][internalTickIndex] = priorBTTeamPlayerData.velocity;
                        if (isnan(priorBTTeamPlayerData.curFootPos.x) || isnan(priorBTTeamPlayerData.curFootPos.y) || isnan(priorBTTeamPlayerData.curFootPos.z) ) {
                            std::cout << "found nan" << std::endl;
                        }
                    }
                }
                columnData[columnIndex].priorFootPosValid[j][internalTickIndex] = setPlayerPriorFootPos;
            }
        }

        /*
        if (internalTickIndex == 1195) {
            std::cout << "T 0 player id " << columnTData[0].playerId[internalTickIndex] << " pos " << t0Pos.toCSV() << " area id " << areaId
                << " distance to BSite order 2 " << columnTData[0].distanceToNearestBOrderNavArea[0][internalTickIndex] << std::endl;
            for (size_t k = 0; k < bOrders[2].places.size(); k++) {
                std::cout << "place " << bOrders[2].places[k] << " " << distanceToPlaces.places[bOrders[2].places[k]]
                    << " distance " << distanceToPlaces.getClosestDistance(areaIndex, bOrders[2].places[k]) << std::endl;
            }
        }
         */
    }

    /*
    void TeamFeatureStoreResult::computeOrderACausalLabels(int64_t internalTickId, CircularBuffer<int64_t> & futureTracker,
                                                           array<ColumnPlayerData, maxEnemies> & columnData,
                                                           ACausalTimingOption timingOption) {
        for (size_t playerColumn = 0; playerColumn < maxEnemies; playerColumn++) {
            if (columnData[playerColumn].playerId[internalTickId] == INVALID_ID) {
                continue;
            }
            // clear out values for current tick
            for (size_t orderPerSite = 0; orderPerSite < num_orders_per_site; orderPerSite++) {
                if (timingOption == ACausalTimingOption::s6) {
                    columnData[playerColumn].distributionNearestAOrders[orderPerSite][internalTickId] = 0;
                    columnData[playerColumn].distributionNearestBOrders[orderPerSite][internalTickId] = 0;
                }
                else if (timingOption == ACausalTimingOption::s15) {
                    //columnData[playerColumn].distributionNearestAOrders15s[orderPerSite][internalTickId] = 0;
                    //columnData[playerColumn].distributionNearestBOrders15s[orderPerSite][internalTickId] = 0;
                }
                else {
                    //columnData[playerColumn].distributionNearestAOrders30s[orderPerSite][internalTickId] = 0;
                    //columnData[playerColumn].distributionNearestBOrders30s[orderPerSite][internalTickId] = 0;
                }
            }
            // want all points where alive, and accounting for ties
            size_t numPointsInDistribution = 0;
            for (int64_t futureTickIndex = 0; futureTickIndex < futureTracker.getCurSize(); futureTickIndex++) {
                int64_t futureTick = futureTracker.fromOldest(futureTickIndex);
                if (futureTick != internalTickId &&
                    columnData[playerColumn].playerId[internalTickId] == columnData[playerColumn].playerId[futureTick]) {
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
                            columnData[playerColumn].distributionNearestAOrders[orderPerSite][internalTickId]++;
                        }
                        else if (timingOption == ACausalTimingOption::s15) {
                            //columnData[playerColumn].distributionNearestAOrders15s[orderPerSite][internalTickId]++;
                        }
                        else {
                            //columnData[playerColumn].distributionNearestAOrders30s[orderPerSite][internalTickId]++;
                        }
                    }
                    for (const auto & orderPerSite : minDistanceBOrders) {
                        if (timingOption == ACausalTimingOption::s6) {
                            columnData[playerColumn].distributionNearestBOrders[orderPerSite][internalTickId]++;
                        }
                        else if (timingOption == ACausalTimingOption::s15) {
                            //columnData[playerColumn].distributionNearestBOrders15s[orderPerSite][internalTickId]++;
                        }
                        else {
                            //columnData[playerColumn].distributionNearestBOrders30s[orderPerSite][internalTickId]++;
                        }
                    }
                }
            }
            if (numPointsInDistribution != 0) {
                for (size_t orderPerSite = 0; orderPerSite < num_orders_per_site; orderPerSite++) {
                    if (timingOption == ACausalTimingOption::s6) {
                        columnData[playerColumn].distributionNearestAOrders[orderPerSite][internalTickId] /=
                            static_cast<double>(numPointsInDistribution);
                        columnData[playerColumn].distributionNearestBOrders[orderPerSite][internalTickId] /=
                            static_cast<double>(numPointsInDistribution);
                    }
                    else if (timingOption == ACausalTimingOption::s15) {
                        //columnData[playerColumn].distributionNearestAOrders15s[orderPerSite][internalTickId] /=
                        //    static_cast<double>(numPointsInDistribution);
                        //columnData[playerColumn].distributionNearestBOrders15s[orderPerSite][internalTickId] /=
                        //    static_cast<double>(numPointsInDistribution);
                    }
                    else {
                        //columnData[playerColumn].distributionNearestAOrders30s[orderPerSite][internalTickId] /=
                        //    static_cast<double>(numPointsInDistribution);
                        //columnData[playerColumn].distributionNearestBOrders30s[orderPerSite][internalTickId] /=
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
                bool inWindow = secondsBetweenTicks(ticks, tickRates, internalIdToTickId[curTick], internalIdToTickId[futureTick]) >=
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
                / *
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
                 * /
                int64_t curPlayerId = columnData[playerColumn].playerId[curTick];
                for (const PlaceIndex badCurPlace : {18, 3, 5, 20, 25}) {
                    for (const PlaceIndex badNextPlace : {11, 0})
                    if (curPlace == badCurPlace && columnData[playerColumn].distributionNearestPlace[badNextPlace][curTick] > 0) {
                        double distanceBetweenCurAndNextPlace =
                                distanceToPlacesResult.getClosestDistance(distanceToPlacesResult.places[curPlace], distanceToPlacesResult.places[badNextPlace], navFile);
                        std::cout << games.demoFile[curGame] << " bad tick reached (" << curPlace << ","
                                  << distanceToPlacesResult.places[curPlace] << ") to (" << badNextPlace << ","
                                  << distanceToPlacesResult.places[badNextPlace] << ") in under 2 seconds "
                                  << " for player (" << curPlayerId << "," << players.name[curPlayerId + players.idOffset] << ") on tick id "
                                  << internalIdToTickId[curTick] << " and game tick " << ticks.gameTickNumber[internalIdToTickId[curTick]] << " distance " << distanceBetweenCurAndNextPlace << std::endl;
                        for (int64_t futureTickIndex = 0; futureTickIndex < futureTracker.getCurSize(); futureTickIndex++) {
                            int64_t futureTick = futureTracker.fromOldest(futureTickIndex);
                            bool inWindow = secondsBetweenTicks(ticks, tickRates, internalIdToTickId[curTick], internalIdToTickId[futureTick]) >=
                                            futureSecondsThreshold;
                            if (futureTick != curTick &&
                                columnData[playerColumn].playerId[curTick] ==
                                columnData[playerColumn].playerId[futureTick] &&
                                inWindow && columnData[playerColumn].curPlace[badNextPlace][futureTick]) {
                                std::cout << "future tick " << internalIdToTickId[futureTick] << std::endl;
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
                bool inWindow = secondsBetweenTicks(ticks, tickRates, internalIdToTickId[curTick], internalIdToTickId[futureTick]) >=
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
    */

    int getDeltaPosFlatIndex(Vec3 pos, AABB placeAABB, bool jumping, bool falling) {
        double xPct = std::max(0., std::min(1., (pos.x - placeAABB.min.x) / (placeAABB.max.x - placeAABB.min.x)));
        double yPct = std::max(0., std::min(1., (pos.y - placeAABB.min.y) / (placeAABB.max.y - placeAABB.min.y)));
        int xValue = static_cast<int>(xPct * delta_pos_grid_num_cells_per_xy_dim);
        if (xValue == delta_pos_grid_num_cells_per_xy_dim) {
            xValue--;
        }
        int yValue = static_cast<int>(yPct * delta_pos_grid_num_cells_per_xy_dim);
        if (yValue == delta_pos_grid_num_cells_per_xy_dim) {
            yValue--;
        }
        int zValue = 1;
        if (jumping) {
            zValue = 2;
        }
        else if (falling) {
            zValue = 0;
        }
        return xValue + yValue * delta_pos_grid_num_cells_per_xy_dim + zValue * delta_pos_grid_num_xy_cells_per_z_change;
    }

    //double max_z_delta = 0;

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
            // if jumping, look twice as far in future
            bool jumping = false, falling = false;
            for (int64_t i = 0; i < futureTracker.getCurSize(); i++) {
                if (columnData[playerColumn].velocity[futureTracker.fromNewest(i)].z > 10.) {
                    jumping = true;
                }
                if (columnData[playerColumn].velocity[futureTracker.fromNewest(i)].z < -10.) {
                    falling = true;
                }
            }
            falling = falling && !jumping;

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
            //max_z_delta = std::max(max_z_delta, std::abs(columnData[playerColumn].footPos[futureTickIndex].z - curFootPos.z));
            int deltaPosIndex = getDeltaPosFlatIndex(columnData[playerColumn].footPos[futureTickIndex], deltaPosRange,
                                                     jumping, falling);
            /*
            // if jumping and standing still in xy, look twice as far in future
            if (deltaPosIndex == 12 && jumping) {
                futureTickIndex = jumpFutureTracker.fromOldest();
                deltaPosIndex = getDeltaPosFlatIndex(columnData[playerColumn].footPos[futureTickIndex], deltaPosRange);
            }
             */
            columnData[playerColumn].deltaPos[deltaPosIndex][curTick] = true;
        }
    }

    void TeamFeatureStoreResult::computeAcausalLabels(const Games & games, const Rounds & rounds,
                                                      const Ticks & ticks,
                                                      const Players &,
                                                      const DistanceToPlacesResult &,
                                                      const nav_mesh::nav_file &,
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
                bothSidesTicks0_5sFutureTracker(20), bothSidesTicks1_5sFutureTracker(30);
            //, ticks15sFutureTracker(15), ticks30sFutureTracker(30);
            for (int64_t unmodifiedTickIndex = rounds.ticksPerRound[roundIndex].maxId;
                 unmodifiedTickIndex >= rounds.ticksPerRound[roundIndex].minId; unmodifiedTickIndex--) {
                int64_t tickIndex = tickIdToInternalId[unmodifiedTickIndex];
                if (tickIndex == INVALID_ID) {
                    continue;
                }
                gameId[tickIndex] = gameIndex;
                gameTickNumber[tickIndex] = ticks.gameTickNumber[unmodifiedTickIndex];
                roundNumber[tickIndex] = rounds.roundNumber[roundIndex];
                freezeTimeEnded[tickIndex] = rounds.freezeTimeEnd[roundIndex] <= unmodifiedTickIndex;
                retakeSaveRoundTick[tickIndex] = keyRetakeEvents.ctAliveAfterExplosion[unmodifiedTickIndex] ||
                        keyRetakeEvents.ctAliveAfterExplosion[unmodifiedTickIndex];
                testName[tickIndex] = keyRetakeEvents.roundTestName[roundIndex];
                testSuccess[tickIndex] = keyRetakeEvents.roundHasCompleteTest[roundIndex];
                baiting[tickIndex] = keyRetakeEvents.roundBaiters[roundIndex];
                // add a new tick every second
                if (ticks1sFutureTracker.isEmpty() ||
                    secondsBetweenTicks(ticks, tickRates, internalIdToTickId[tickIndex], internalIdToTickId[ticks1sFutureTracker.fromNewest()]) >= 0.25) {
                    ticks1sFutureTracker.enqueue(tickIndex);
                }
                if (ticks2sFutureTracker.isEmpty() ||
                    secondsBetweenTicks(ticks, tickRates, internalIdToTickId[tickIndex], internalIdToTickId[ticks2sFutureTracker.fromNewest()]) >= 0.5) {
                    ticks2sFutureTracker.enqueue(tickIndex);
                }
                if (ticks6sFutureTracker.isEmpty() ||
                    secondsBetweenTicks(ticks, tickRates, internalIdToTickId[tickIndex], internalIdToTickId[ticks6sFutureTracker.fromNewest()]) >= 1.) {
                    ticks6sFutureTracker.enqueue(tickIndex);
                }
                bothSidesTicks0_5sFutureTracker.enqueue(tickIndex);
                while (secondsBetweenTicks(ticks, tickRates, internalIdToTickId[tickIndex], internalIdToTickId[bothSidesTicks0_5sFutureTracker.fromOldest()]) > 0.5) {
                    bothSidesTicks0_5sFutureTracker.dequeue();
                }
                bothSidesTicks1_5sFutureTracker.enqueue(tickIndex);
                while (secondsBetweenTicks(ticks, tickRates, internalIdToTickId[tickIndex], internalIdToTickId[bothSidesTicks1_5sFutureTracker.fromOldest()]) > 1.5) {
                    bothSidesTicks1_5sFutureTracker.dequeue();
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
                if (ticks.roundId[internalIdToTickId[ticks1sFutureTracker.fromOldest()]] != ticks.roundId[internalIdToTickId[tickIndex]]) {
                    std::cout << "round id mismatch cur tick " << internalIdToTickId[tickIndex] << " future tick " << internalIdToTickId[ticks1sFutureTracker.fromOldest()] << std::endl;
                    std::raise(SIGINT);
                }
                /*
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
                 */
                computeDeltaPosACausalLabels(tickIndex, bothSidesTicks0_5sFutureTracker, columnCTData);
                computeDeltaPosACausalLabels(tickIndex, bothSidesTicks0_5sFutureTracker, columnTData);
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
                if (ticks.gameTickNumber[unmodifiedTickIndex] == 236466) {
                    std::cout << "good round " << roundIndex << std::endl;
                    std::raise(SIGINT);
                }
                 */
            }
            roundsProcessed++;
            printProgress(roundsProcessed, rounds.size);
        }
        //std::cout << "max z delta " << max_z_delta << std::endl;
    }

    void TeamFeatureStoreResult::toHDF5Inner(HighFive::File & file) {
        HighFive::DataSetCreateProps hdf5FlatCreateProps;
        //hdf5FlatCreateProps.add(HighFive::Deflate(6));
        //hdf5FlatCreateProps.add(HighFive::Chunking(roundId.size()));

        file.createDataSet("/data/game id", gameId, hdf5FlatCreateProps);
        file.createDataSet("/data/round id", roundId, hdf5FlatCreateProps);
        file.createDataSet("/data/tick id", tickId, hdf5FlatCreateProps);
        file.createDataSet("/data/game tick number", gameTickNumber, hdf5FlatCreateProps);
        file.createDataSet("/data/round number", roundNumber, hdf5FlatCreateProps);
        file.createDataSet("/data/valid", valid, hdf5FlatCreateProps);
        file.createDataSet("/data/freeze time ended", freezeTimeEnded, hdf5FlatCreateProps);
        file.createDataSet("/data/retake save round tick", retakeSaveRoundTick, hdf5FlatCreateProps);
        file.createDataSet("/data/test name", testName, hdf5FlatCreateProps);
        file.createDataSet("/data/test success", testSuccess, hdf5FlatCreateProps);
        file.createDataSet("/data/baiting", baiting, hdf5FlatCreateProps);
        file.createDataSet("/data/c4 status", vectorOfEnumsToVectorOfInts(c4Status), hdf5FlatCreateProps);
        file.createDataSet("/data/c4 planted a", c4PlantA, hdf5FlatCreateProps);
        file.createDataSet("/data/c4 planted b", c4PlantB, hdf5FlatCreateProps);
        file.createDataSet("/data/c4 not planted", c4NotPlanted, hdf5FlatCreateProps);
        file.createDataSet("/data/c4 ticks since plant", c4TicksSincePlant, hdf5FlatCreateProps);
        for (size_t c4TimerBucketIndex = 0; c4TimerBucketIndex < num_c4_timer_buckets; c4TimerBucketIndex++) {
            file.createDataSet("/data/c4 timer bucketed " + std::to_string(c4TimerBucketIndex), c4TimerBucketed[c4TimerBucketIndex], hdf5FlatCreateProps);
        }
        saveVec3VectorToHDF5(c4Pos, file, "c4 pos", hdf5FlatCreateProps);
        file.createDataSet("/data/c4 distance to a site", c4DistanceToASite, hdf5FlatCreateProps);
        file.createDataSet("/data/c4 distance to b site", c4DistanceToBSite, hdf5FlatCreateProps);
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
                file.createDataSet("/data/alive " + columnTeam + " " + iStr,
                                   columnData[columnPlayer].alive, hdf5FlatCreateProps);
                file.createDataSet("/data/player ctTeam " + columnTeam + " " + iStr,
                                   columnData[columnPlayer].ctTeam, hdf5FlatCreateProps);
                saveVec3VectorToHDF5(columnData[columnPlayer].footPos, file,
                                     "player pos " + columnTeam + " " + iStr, hdf5FlatCreateProps);
                saveVec3VectorToHDF5(columnData[columnPlayer].velocity, file,
                                     "player velocity " + columnTeam + " " + iStr, hdf5FlatCreateProps);
                for (int priorTick = 0; priorTick < num_prior_ticks; priorTick++) {
                    saveVec3VectorToHDF5(columnData[columnPlayer].priorFootPos[priorTick], file,
                                         "player pos " + columnTeam + " " + iStr + " t-" + std::to_string(priorTick+1),
                                         hdf5FlatCreateProps);
                    saveVec3VectorToHDF5(columnData[columnPlayer].priorVelocity[priorTick], file,
                                         "player velocity " + columnTeam + " " + iStr + " t-" + std::to_string(priorTick+1),
                                         hdf5FlatCreateProps);
                    file.createDataSet("/data/player history valid " + columnTeam + " " + iStr + " t-" + std::to_string(priorTick+1),
                                       columnData[columnPlayer].priorFootPosValid[priorTick], hdf5FlatCreateProps);
                }
                vector<string> deltaPosNames;
                for (size_t deltaPosIndex = 0; deltaPosIndex < delta_pos_grid_num_cells; deltaPosIndex++) {
                    string deltaPosIndexStr = std::to_string(deltaPosIndex);
                    file.createDataSet("/data/delta pos " + deltaPosIndexStr + " " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].deltaPos[deltaPosIndex], hdf5FlatCreateProps);
                }
            }
        }
    }

    void TeamFeatureStoreResult::load(const std::string &filePath) {
        HighFive::File file(filePath, HighFive::File::ReadOnly);
        fileName = std::filesystem::path(filePath).filename();

        roundId = file.getDataSet("/data/round id").read<std::vector<int64_t>>();
        roundNumber = file.getDataSet("/data/round number").read<std::vector<int64_t>>();
        tickId = file.getDataSet("/data/tick id").read<std::vector<int64_t>>();
        gameTickNumber = file.getDataSet("/data/game tick number").read<std::vector<int64_t>>();
        valid = file.getDataSet("/data/valid").read<std::vector<bool>>();
        freezeTimeEnded = file.getDataSet("/data/freeze time ended").read<std::vector<bool>>();
        retakeSaveRoundTick = file.getDataSet("/data/retake save round tick").read<std::vector<bool>>();
        testName = file.getDataSet("/data/test name").read<std::vector<string>>();
        testSuccess = file.getDataSet("/data/test success").read<std::vector<bool>>();
        baiting = file.getDataSet("/data/baiting").read<std::vector<bool>>();
        loadVectorOfEnums(file, "/data/c4 status", c4Status);
        c4PlantA = file.getDataSet("/data/c4 planted a").read<std::vector<bool>>();
        c4PlantB = file.getDataSet("/data/c4 planted b").read<std::vector<bool>>();
        c4NotPlanted = file.getDataSet("/data/c4 not planted").read<std::vector<bool>>();
        c4TicksSincePlant = file.getDataSet("/data/c4 ticks since plant").read<std::vector<int64_t>>();
        for (size_t c4TimerBucketIndex = 0; c4TimerBucketIndex < num_c4_timer_buckets; c4TimerBucketIndex++) {
            c4TimerBucketed[c4TimerBucketIndex] = file.getDataSet("/data/c4 timer bucketed " + std::to_string(c4TimerBucketIndex)).read<std::vector<bool>>();
        }
        loadVec3VectorFromHDF5(c4Pos, file, "c4 pos");
        c4DistanceToASite = file.getDataSet("/data/c4 distance to a site").read<std::vector<float>>();
        c4DistanceToBSite = file.getDataSet("/data/c4 distance to b site").read<std::vector<float>>();
        for (size_t columnDataIndex = 0; columnDataIndex < getAllColumnData().size(); columnDataIndex++) {
            array<ColumnPlayerData, maxEnemies> &columnData = getAllColumnData()[columnDataIndex];
            string columnTeam = allColumnDataTeam[columnDataIndex];
            for (size_t columnPlayer = 0; columnPlayer < columnData.size(); columnPlayer++) {
                string iStr = std::to_string(columnPlayer);
                columnData[columnPlayer].playerId = file.getDataSet(
                        "/data/player id " + columnTeam + " " + iStr).read<std::vector<int64_t>>();
                for (int indexOnTeam = 0; indexOnTeam < maxEnemies; indexOnTeam++) {
                    columnData[columnPlayer].indexOnTeam[indexOnTeam] = file.getDataSet(
                            "/data/player index on team " + std::to_string(indexOnTeam) + " " + columnTeam + " " + iStr).read<std::vector<bool>>();
                }
                columnData[columnPlayer].alive = file.getDataSet("/data/alive " + columnTeam + " " + iStr).read<std::vector<bool>>();
                columnData[columnPlayer].ctTeam = file.getDataSet("/data/player ctTeam " + columnTeam + " " + iStr).read<std::vector<bool>>();
                loadVec3VectorFromHDF5(columnData[columnPlayer].footPos, file, "player pos " + columnTeam + " " + iStr);
                loadVec3VectorFromHDF5(columnData[columnPlayer].velocity, file, "player velocity " + columnTeam + " " + iStr);
                for (int priorTick = 0; priorTick < num_prior_ticks; priorTick++) {
                    loadVec3VectorFromHDF5(columnData[columnPlayer].priorFootPos[priorTick], file,
                                           "player pos " + columnTeam + " " + iStr + " t-" + std::to_string(priorTick + 1));
                    loadVec3VectorFromHDF5(columnData[columnPlayer].priorVelocity[priorTick], file,
                                           "player velocity " + columnTeam + " " + iStr + " t-" + std::to_string(priorTick + 1));
                    columnData[columnPlayer].priorFootPosValid[priorTick] =
                            file.getDataSet("/data/player history valid " + columnTeam + " " + iStr + " t-" + std::to_string(priorTick + 1)).read<std::vector<bool>>();
                }
                vector<string> deltaPosNames;
                for (size_t deltaPosIndex = 0; deltaPosIndex < delta_pos_grid_num_cells; deltaPosIndex++) {
                    string deltaPosIndexStr = std::to_string(deltaPosIndex);
                    columnData[columnPlayer].deltaPos[deltaPosIndex] =
                            file.getDataSet("/data/delta pos " + deltaPosIndexStr + " " + columnTeam + " " + iStr).read<std::vector<bool>>();
                }
            }
        }
        size = static_cast<int64_t>(valid.size());
    }
}
