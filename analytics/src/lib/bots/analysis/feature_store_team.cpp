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
        c4AreaIndex.resize(size, INVALID_ID);
        c4Status.resize(size, C4Status::NotPlanted);
        c4PlantA.resize(size, false);
        c4PlantB.resize(size, false);
        c4NotPlanted.resize(size, false);
        c4TicksSincePlant.resize(size, INVALID_ID);
        c4TimeLeftPercent.resize(size, 1.);
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
        for (int i = 0; i < max_enemies; i++) {
            for (int j = 0; j < max_enemies; j++) {
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
            columnTData[i].areaIndex.resize(size, INVALID_ID);
            columnCTData[i].areaIndex.resize(size, INVALID_ID);
            columnTData[i].decreaseDistanceToC4Over5s.resize(size, false);
            columnCTData[i].decreaseDistanceToC4Over5s.resize(size, false);
            columnTData[i].decreaseDistanceToC4Over10s.resize(size, false);
            columnCTData[i].decreaseDistanceToC4Over10s.resize(size, false);
            columnTData[i].decreaseDistanceToC4Over20s.resize(size, false);
            columnCTData[i].decreaseDistanceToC4Over20s.resize(size, false);
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
            for (int j = 0; j < delta_pos_grid_num_cells; j++) {
                columnTData[i].deltaPos[j].resize(size, false);
                columnCTData[i].deltaPos[j].resize(size, false);
            }
             */
            columnTData[i].weaponId.resize(size, EngineWeaponId::None);
            columnCTData[i].weaponId.resize(size, EngineWeaponId::None);
            columnTData[i].scoped.resize(size, false);
            columnCTData[i].scoped.resize(size, false);
            columnTData[i].airborne.resize(size, false);
            columnCTData[i].airborne.resize(size, false);
            columnTData[i].walking.resize(size, false);
            columnCTData[i].walking.resize(size, false);
            columnTData[i].ducking.resize(size, false);
            columnCTData[i].ducking.resize(size, false);
            for (int j = 0; j < weapon_speed::num_radial_bins; j++) {
                columnTData[i].radialVel[j].resize(size, false);
                columnCTData[i].radialVel[j].resize(size, false);
            }
            for (int j = 0; j < num_future_ticks; j++) {
                for (int k = 0; k < weapon_speed::num_radial_bins; k++) {
                    columnTData[i].futureRadialVel[j][k].resize(size, false);
                    columnCTData[i].futureRadialVel[j][k].resize(size, false);
                }
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
            c4AreaIndex[rowIndex] = INVALID_ID;
            c4Status[rowIndex] = C4Status::NotPlanted;
            c4PlantA[rowIndex] = false;
            c4PlantB[rowIndex] = false;
            c4NotPlanted[rowIndex] = false;
            c4TicksSincePlant[rowIndex] = INVALID_ID;
            c4TimeLeftPercent[rowIndex] = 1.;
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
            for (int i = 0; i < max_enemies; i++) {
                for (int j = 0; j < max_enemies; j++) {
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
                columnTData[i].areaIndex[rowIndex] = INVALID_ID;
                columnCTData[i].areaIndex[rowIndex] = INVALID_ID;
                columnTData[i].decreaseDistanceToC4Over5s[rowIndex] = false;
                columnCTData[i].decreaseDistanceToC4Over5s[rowIndex] = false;
                columnTData[i].decreaseDistanceToC4Over10s[rowIndex] = false;
                columnCTData[i].decreaseDistanceToC4Over10s[rowIndex] = false;
                columnTData[i].decreaseDistanceToC4Over20s[rowIndex] = false;
                columnCTData[i].decreaseDistanceToC4Over20s[rowIndex] = false;
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
                for (int j = 0; j < delta_pos_grid_num_cells; j++) {
                    columnTData[i].deltaPos[j][rowIndex] = false;
                    columnCTData[i].deltaPos[j][rowIndex] = false;
                }
                 */
                columnTData[i].weaponId[rowIndex] = EngineWeaponId::None;
                columnCTData[i].weaponId[rowIndex] = EngineWeaponId::None;
                columnTData[i].scoped[rowIndex] = false;
                columnCTData[i].scoped[rowIndex] = false;
                columnTData[i].airborne[rowIndex] = false;
                columnCTData[i].airborne[rowIndex] = false;
                columnTData[i].walking[rowIndex] = false;
                columnCTData[i].walking[rowIndex] = false;
                columnTData[i].ducking[rowIndex] = false;
                columnCTData[i].ducking[rowIndex] = false;
                for (int j = 0; j < weapon_speed::num_radial_bins; j++) {
                    columnTData[i].radialVel[j][rowIndex] = false;
                    columnCTData[i].radialVel[j][rowIndex] = false;
                }
                for (int j = 0; j < num_future_ticks; j++) {
                    for (int k = 0; k < weapon_speed::num_radial_bins; k++) {
                        columnTData[i].futureRadialVel[j][k][rowIndex] = false;
                        columnCTData[i].futureRadialVel[j][k][rowIndex] = false;
                    }
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

    bool TeamFeatureStoreResult::commitTeamRow(FeatureStorePreCommitBuffer & buffer,
                                               const DistanceToPlacesResult & distanceToPlaces,
                                               const nav_mesh::nav_file & navFile,
                                               int64_t roundIndex, int64_t tickIndex) {
        int64_t internalTickIndex = tickIdToInternalId[tickIndex];
        if (internalTickIndex == INVALID_ID) {
            return false;
        }

        roundId[internalTickIndex] = roundIndex;
        tickId[internalTickIndex] = internalTickIndex;
        valid[internalTickIndex] = !buffer.btTeamPlayerData.empty();

        if (buffer.c4MapData.c4Planted) {
            c4AreaIndex[internalTickIndex] = buffer.c4MapData.c4AreaIndex;
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
            c4AreaIndex[internalTickIndex] = INVALID_ID;
            c4Status[internalTickIndex] = C4Status::NotPlanted;
            c4NotPlanted[internalTickIndex] = true;
        }
        c4TicksSincePlant[internalTickIndex] = buffer.c4MapData.ticksSincePlant;
        // TODO: adjust for different tick rates
        // need to clamp as there's a period after timer finishes before explosion (when bomb light turns white)
        c4TimeLeftPercent[internalTickIndex] =
                std::clamp(1.f - static_cast<float>(c4TicksSincePlant[internalTickIndex] / 128.f) / c4_max_time_seconds,
                           0.f, 1.f);
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
            columnData[columnIndex].areaIndex[internalTickIndex] = btTeamPlayerData.curAreaIndex;
            columnData[columnIndex].weaponId[internalTickIndex] = btTeamPlayerData.weaponId;
            columnData[columnIndex].scoped[internalTickIndex] = btTeamPlayerData.scoped;
            columnData[columnIndex].airborne[internalTickIndex] = btTeamPlayerData.airborne;
            columnData[columnIndex].walking[internalTickIndex] = btTeamPlayerData.walking;
            columnData[columnIndex].ducking[internalTickIndex] = btTeamPlayerData.ducking;
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
        return true;
    }

    void TeamFeatureStoreResult::computeDeltaPosACausalLabels(int64_t curTick, CircularBuffer<int64_t> & futureTracker,
                                                              array<ColumnPlayerData,max_enemies> & columnData) {
        if (futureTracker.isEmpty()) {
            std::cout << "delta pos acausal label with no future ticks" << std::endl;
            std::raise(SIGINT);
        }
        // skip if not enough history in the future
        if (futureTracker.getCurSize() < 2) {
            return;
        }
        for (size_t playerColumn = 0; playerColumn < max_enemies; playerColumn++) {
            if (columnData[playerColumn].playerId[curTick] == INVALID_ID) {
                continue;
            }
            // clear out values for current tick
            /*
            for (size_t deltaPosGridIndex = 0; deltaPosGridIndex < delta_pos_grid_num_cells; deltaPosGridIndex++) {
                columnData[playerColumn].deltaPos[deltaPosGridIndex][curTick] = false;
            }
             */
            for (int radialVelIndex = 0; radialVelIndex < weapon_speed::num_radial_bins; radialVelIndex++) {
                columnData[playerColumn].radialVel[radialVelIndex][curTick] = false;
            }

            // ignore players who aren't alive far enough in the future
            // look at next tick;
            int64_t futureTickIndex = futureTracker.fromNewest(1);
            if (columnData[playerColumn].playerId[futureTickIndex] != columnData[playerColumn].playerId[curTick]) {
                continue;
            }

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
            /*
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
             */
            //max_z_delta = std::max(max_z_delta, std::abs(columnData[playerColumn].footPos[futureTickIndex].z - curFootPos.z));
            /*
            int deltaPosIndex = getDeltaPosFlatIndex(columnData[playerColumn].footPos[futureTickIndex], deltaPosRange,
                                                     jumping, falling);
                                                     */
            weapon_speed::StatureOptions statureOption = weapon_speed::StatureOptions::Standing;
            if (columnData[playerColumn].ducking[curTick]) {
                statureOption = weapon_speed::StatureOptions::Ducking;
            }
            else if (columnData[playerColumn].walking[curTick]) {
                statureOption = weapon_speed::StatureOptions::Walking;
            }
            weapon_speed::MovementStatus movementStatus(columnData[playerColumn].weaponId[curTick],
                                                        columnData[playerColumn].velocity[curTick],
                                                        columnData[playerColumn].velocity[futureTickIndex],
                                                        statureOption,
                                                        columnData[playerColumn].scoped[curTick],
                                                        columnData[playerColumn].airborne[futureTickIndex],
                                                        jumping, falling);
            columnData[playerColumn].radialVel[movementStatus.toRadialMovementBin()][curTick] = true;
        }
    }

    void TeamFeatureStoreResult::computeFutureDeltaPosACausalLabels(int64_t curTick, CircularBuffer<int64_t> & futureTracker0_5,
                                                                    CircularBuffer<int64_t> & futureTracker1_0,
                                                                    array<ColumnPlayerData,max_enemies> & columnData,
                                                                    const TickRates & tickRates) {
        if (futureTracker1_0.isEmpty() || futureTracker0_5.isEmpty()) {
            std::cout << "future delta pos acausal label with no future ticks" << std::endl;
            std::raise(SIGINT);
        }

        // if enough distance between present and future, then fill in
        // otherwise disable future tick
        if (secondsBetweenTicks(tickRates, gameTickNumber[curTick], gameTickNumber[futureTracker1_0.fromOldest()]) < 0.8) {
            return;
        }

        for (size_t playerColumn = 0; playerColumn < max_enemies; playerColumn++) {
            if (columnData[playerColumn].playerId[curTick] == INVALID_ID) {
                continue;
            }

            // ignore players who aren't alive far enough in the future
            // look at next tick;
            int64_t futureTickIndex0_5 = futureTracker0_5.fromOldest(),
                futureTickIndex1_0 = futureTracker1_0.fromOldest();
            if (columnData[playerColumn].playerId[futureTickIndex1_0] != columnData[playerColumn].playerId[curTick]) {
                continue;
            }

            for (int radialVelIndex = 0; radialVelIndex < weapon_speed::num_radial_bins; radialVelIndex++) {
                columnData[playerColumn].futureRadialVel[0][radialVelIndex][curTick] =
                        columnData[playerColumn].radialVel[radialVelIndex][futureTickIndex0_5];
                columnData[playerColumn].futureRadialVel[1][radialVelIndex][curTick] =
                        columnData[playerColumn].radialVel[radialVelIndex][futureTickIndex1_0];
            }
        }
    }

    void TeamFeatureStoreResult::removePartialACausalLabels(int64_t curTick,
                                                            array<ColumnPlayerData,max_enemies> & columnData) {
        for (size_t playerColumn = 0; playerColumn < max_enemies; playerColumn++) {
            // valid if one output set (assume never more than 1 set)
            bool curPlayerTickValid = false, futurePlayerTickValid0_5 = false, futurePlayerTickValid1_0 = false;
            for (int radialVelIndex = 0; radialVelIndex < weapon_speed::num_radial_bins; radialVelIndex++) {
                curPlayerTickValid = curPlayerTickValid || columnData[playerColumn].radialVel[radialVelIndex][curTick];
                futurePlayerTickValid0_5 = futurePlayerTickValid0_5 ||
                        columnData[playerColumn].futureRadialVel[0][radialVelIndex][curTick];
                futurePlayerTickValid1_0 = futurePlayerTickValid0_5 ||
                                           columnData[playerColumn].futureRadialVel[1][radialVelIndex][curTick];
            }

            // disable all labels if any one isn't set
            if (!curPlayerTickValid || !futurePlayerTickValid0_5 || !futurePlayerTickValid1_0) {
                for (int radialVelIndex = 0; radialVelIndex < weapon_speed::num_radial_bins; radialVelIndex++) {
                    columnData[playerColumn].radialVel[radialVelIndex][curTick] = false;
                    columnData[playerColumn].futureRadialVel[0][radialVelIndex][curTick] = false;
                    columnData[playerColumn].futureRadialVel[1][radialVelIndex][curTick] = false;
                }
            }
        }
    }

    constexpr double c4_distance_threshold = 1000, c4_delta_distance_threshold = 150;
    void TeamFeatureStoreResult::computeDecreaseDistanceToC4(
            int64_t curTick, CircularBuffer<int64_t> &futureTracker,
            array<csknow::feature_store::TeamFeatureStoreResult::ColumnPlayerData, max_enemies> &columnData,
            DecreaseTimingOption decreaseTimingOption, const ReachableResult & reachableResult) {
        // skip if not enough history in the future
        if (futureTracker.getCurSize() < 2) {
            return;
        }
        for (size_t playerColumn = 0; playerColumn < max_enemies; playerColumn++) {
            if (columnData[playerColumn].playerId[curTick] == INVALID_ID) {
                continue;
            }

            // find farthest forward where player alive

            int64_t futureTickIndex = INVALID_ID;
            // don't look at current tick, need at least 1 tick in the future
            for (int64_t i = 0; i < futureTracker.getCurSize() - 1; i++) {
                futureTickIndex = futureTracker.fromOldest(i);
                if (columnData[playerColumn].playerId[futureTickIndex] == columnData[playerColumn].playerId[curTick]) {
                    break;
                }
            }
            // default to false if this is last tick, this won't be used in training since no movement data
            if (futureTickIndex == INVALID_ID) {
                continue;
            }

            if (futureTickIndex < curTick) {
                std::cout << "decreasing distance to c4 future tick in past" << std::endl;
                std::raise(SIGINT);
            }

            double curDistanceToC4 = reachableResult.getDistance(columnData[playerColumn].areaIndex[curTick],
                                                                 c4AreaIndex[curTick]);
            double futureDistanceToC4 = reachableResult.getDistance(columnData[playerColumn].areaIndex[futureTickIndex],
                                                                    c4AreaIndex[futureTickIndex]);

            bool decreaseDistance = (curDistanceToC4 < c4_distance_threshold && futureDistanceToC4 < c4_distance_threshold) ||
                    (futureDistanceToC4 + c4_delta_distance_threshold < curDistanceToC4);
            /*
            if (curTick == 38 && playerColumn == 0) {
                std::cout << "cur tick " << curTick << ", cur distance " << curDistanceToC4 << ", future tick " << futureTickIndex << ", future distance " << futureDistanceToC4 << std::endl;
                std::cout << "hi" << std::endl;
            }
             */
            if (decreaseTimingOption == DecreaseTimingOption::s5) {
                columnData[playerColumn].decreaseDistanceToC4Over5s[curTick] = decreaseDistance;
            }
            else if (decreaseTimingOption == DecreaseTimingOption::s10) {
                columnData[playerColumn].decreaseDistanceToC4Over10s[curTick] = decreaseDistance;
            }
            else if (decreaseTimingOption == DecreaseTimingOption::s20) {
                columnData[playerColumn].decreaseDistanceToC4Over20s[curTick] = decreaseDistance;
            }
        }
    }

    void TeamFeatureStoreResult::computeAcausalLabels(const Games & games, const Rounds & rounds,
                                                      const Ticks & ticks,
                                                      const Players &,
                                                      const DistanceToPlacesResult &,
                                                      const ReachableResult & reachableResult,
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
            CircularBuffer<int64_t> bothSidesTicks0_5sFutureTracker(20), bothSidesTicks1_0sFutureTracker(30),
                bothSidesTicks5sFutureTracker(200), bothSidesTicks10sFutureTracker(200), bothSidesTicks20sFutureTracker(400);
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

                bothSidesTicks0_5sFutureTracker.enqueue(tickIndex);
                while (bothSidesTicks0_5sFutureTracker.getCurSize() > 1 &&
                    secondsBetweenTicks(ticks, tickRates, internalIdToTickId[tickIndex], internalIdToTickId[bothSidesTicks0_5sFutureTracker.fromOldest(1)]) > 0.5) {
                    bothSidesTicks0_5sFutureTracker.dequeue();
                }
                bothSidesTicks1_0sFutureTracker.enqueue(tickIndex);
                while (bothSidesTicks1_0sFutureTracker.getCurSize() > 1 &&
                       secondsBetweenTicks(ticks, tickRates, internalIdToTickId[tickIndex], internalIdToTickId[bothSidesTicks1_0sFutureTracker.fromOldest(1)]) > 0.5) {
                    bothSidesTicks1_0sFutureTracker.dequeue();
                }
                bothSidesTicks5sFutureTracker.enqueue(tickIndex);
                while (bothSidesTicks5sFutureTracker.getCurSize() > 1 &&
                    secondsBetweenTicks(ticks, tickRates, internalIdToTickId[tickIndex], internalIdToTickId[bothSidesTicks5sFutureTracker.fromOldest(1)]) > 5.) {
                    bothSidesTicks5sFutureTracker.dequeue();
                }
                bothSidesTicks10sFutureTracker.enqueue(tickIndex);
                while (bothSidesTicks10sFutureTracker.getCurSize() > 1 &&
                    secondsBetweenTicks(ticks, tickRates, internalIdToTickId[tickIndex], internalIdToTickId[bothSidesTicks10sFutureTracker.fromOldest(1)]) > 10.) {
                    bothSidesTicks10sFutureTracker.dequeue();
                }
                bothSidesTicks20sFutureTracker.enqueue(tickIndex);
                while (bothSidesTicks20sFutureTracker.getCurSize() > 1 &&
                    secondsBetweenTicks(ticks, tickRates, internalIdToTickId[tickIndex], internalIdToTickId[bothSidesTicks20sFutureTracker.fromOldest(1)]) > 20.) {
                    bothSidesTicks20sFutureTracker.dequeue();
                }

                if (ticks.roundId[internalIdToTickId[bothSidesTicks0_5sFutureTracker.fromOldest()]] != ticks.roundId[internalIdToTickId[tickIndex]]) {
                    std::cout << "round id mismatch cur tick " << internalIdToTickId[tickIndex] << " future tick " << internalIdToTickId[bothSidesTicks0_5sFutureTracker.fromOldest()] << std::endl;
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
                computeFutureDeltaPosACausalLabels(tickIndex, bothSidesTicks0_5sFutureTracker,
                                                   bothSidesTicks1_0sFutureTracker, columnCTData, tickRates);
                computeFutureDeltaPosACausalLabels(tickIndex, bothSidesTicks0_5sFutureTracker,
                                                   bothSidesTicks1_0sFutureTracker, columnTData, tickRates);
                computeDecreaseDistanceToC4(tickIndex, bothSidesTicks5sFutureTracker, columnCTData,
                                            DecreaseTimingOption::s5, reachableResult);
                computeDecreaseDistanceToC4(tickIndex, bothSidesTicks5sFutureTracker, columnTData,
                                            DecreaseTimingOption::s5, reachableResult);
                computeDecreaseDistanceToC4(tickIndex, bothSidesTicks10sFutureTracker, columnCTData,
                                            DecreaseTimingOption::s10, reachableResult);
                computeDecreaseDistanceToC4(tickIndex, bothSidesTicks10sFutureTracker, columnTData,
                                            DecreaseTimingOption::s10, reachableResult);
                computeDecreaseDistanceToC4(tickIndex, bothSidesTicks20sFutureTracker, columnCTData,
                                            DecreaseTimingOption::s20, reachableResult);
                computeDecreaseDistanceToC4(tickIndex, bothSidesTicks20sFutureTracker, columnTData,
                                            DecreaseTimingOption::s20, reachableResult);
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
            // after all is done, remove partial labels (where have next step but not 0.5 or 1.0 in future)
            // do after computing all values so can use a value in past, then remove it without impacting past
            for (int64_t unmodifiedTickIndex = rounds.ticksPerRound[roundIndex].minId;
                 unmodifiedTickIndex <= rounds.ticksPerRound[roundIndex].maxId; unmodifiedTickIndex++) {
                int64_t tickIndex = tickIdToInternalId[unmodifiedTickIndex];
                if (tickIndex == INVALID_ID) {
                    continue;
                }
                removePartialACausalLabels(tickIndex, columnCTData);
                removePartialACausalLabels(tickIndex, columnTData);
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
        file.createDataSet("/data/c4 time left percent", c4TimeLeftPercent, hdf5FlatCreateProps);
        for (size_t c4TimerBucketIndex = 0; c4TimerBucketIndex < num_c4_timer_buckets; c4TimerBucketIndex++) {
            file.createDataSet("/data/c4 timer bucketed " + std::to_string(c4TimerBucketIndex), c4TimerBucketed[c4TimerBucketIndex], hdf5FlatCreateProps);
        }
        saveVec3VectorToHDF5(c4Pos, file, "c4 pos", hdf5FlatCreateProps);
        file.createDataSet("/data/c4 distance to a site", c4DistanceToASite, hdf5FlatCreateProps);
        file.createDataSet("/data/c4 distance to b site", c4DistanceToBSite, hdf5FlatCreateProps);
        for (size_t columnDataIndex = 0; columnDataIndex < getAllColumnData().size(); columnDataIndex++) {
            const array<ColumnPlayerData, max_enemies> & columnData = getAllColumnData()[columnDataIndex];
            string columnTeam = allColumnDataTeam[columnDataIndex];
            for (size_t columnPlayer = 0; columnPlayer < columnData.size(); columnPlayer++) {
                string iStr = std::to_string(columnPlayer);
                file.createDataSet("/data/player id " + columnTeam + " " + iStr,
                                   columnData[columnPlayer].playerId, hdf5FlatCreateProps);
                for (int indexOnTeam = 0; indexOnTeam < max_enemies; indexOnTeam++) {
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
                file.createDataSet("/data/player decrease distance to c4 over 5s " + columnTeam + " " + iStr,
                                   columnData[columnPlayer].decreaseDistanceToC4Over5s, hdf5FlatCreateProps);
                file.createDataSet("/data/player decrease distance to c4 over 10s " + columnTeam + " " + iStr,
                                   columnData[columnPlayer].decreaseDistanceToC4Over10s, hdf5FlatCreateProps);
                file.createDataSet("/data/player decrease distance to c4 over 20s " + columnTeam + " " + iStr,
                                   columnData[columnPlayer].decreaseDistanceToC4Over20s, hdf5FlatCreateProps);
                /*
                vector<string> deltaPosNames;
                for (size_t deltaPosIndex = 0; deltaPosIndex < delta_pos_grid_num_cells; deltaPosIndex++) {
                    string deltaPosIndexStr = std::to_string(deltaPosIndex);
                    file.createDataSet("/data/delta pos " + deltaPosIndexStr + " " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].deltaPos[deltaPosIndex], hdf5FlatCreateProps);
                }
                */
                for (int radialVelindex = 0; radialVelindex < weapon_speed::num_radial_bins; radialVelindex++) {
                    file.createDataSet("/data/radial vel " + std::to_string(radialVelindex) + " " + columnTeam + " " + iStr,
                                       columnData[columnPlayer].radialVel[radialVelindex], hdf5FlatCreateProps);
                }
                for (int futureTick = 0; futureTick < num_future_ticks; futureTick++) {
                    for (int radialVelindex = 0; radialVelindex < weapon_speed::num_radial_bins; radialVelindex++) {
                        file.createDataSet("/data/radial vel " + std::to_string(radialVelindex) + " "
                                            + columnTeam + " " + iStr + " t+" + std::to_string(futureTick+1),
                                           columnData[columnPlayer].futureRadialVel[futureTick][radialVelindex], hdf5FlatCreateProps);
                    }
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
        c4TimeLeftPercent = file.getDataSet("/data/c4 time left percent").read<std::vector<float>>();
        for (size_t c4TimerBucketIndex = 0; c4TimerBucketIndex < num_c4_timer_buckets; c4TimerBucketIndex++) {
            c4TimerBucketed[c4TimerBucketIndex] = file.getDataSet("/data/c4 timer bucketed " + std::to_string(c4TimerBucketIndex)).read<std::vector<bool>>();
        }
        loadVec3VectorFromHDF5(c4Pos, file, "c4 pos");
        c4DistanceToASite = file.getDataSet("/data/c4 distance to a site").read<std::vector<float>>();
        c4DistanceToBSite = file.getDataSet("/data/c4 distance to b site").read<std::vector<float>>();
        for (size_t columnDataIndex = 0; columnDataIndex < getAllColumnData().size(); columnDataIndex++) {
            array<ColumnPlayerData, max_enemies> &columnData = getAllColumnData()[columnDataIndex];
            string columnTeam = allColumnDataTeam[columnDataIndex];
            for (size_t columnPlayer = 0; columnPlayer < columnData.size(); columnPlayer++) {
                string iStr = std::to_string(columnPlayer);
                columnData[columnPlayer].playerId = file.getDataSet(
                        "/data/player id " + columnTeam + " " + iStr).read<std::vector<int64_t>>();
                for (int indexOnTeam = 0; indexOnTeam < max_enemies; indexOnTeam++) {
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
                columnData[columnPlayer].decreaseDistanceToC4Over5s =
                        file.getDataSet("/data/player decrease distance to c4 over 5s " + columnTeam + " " + iStr).read<std::vector<bool>>();
                columnData[columnPlayer].decreaseDistanceToC4Over10s =
                        file.getDataSet("/data/player decrease distance to c4 over 10s " + columnTeam + " " + iStr).read<std::vector<bool>>();
                columnData[columnPlayer].decreaseDistanceToC4Over20s =
                        file.getDataSet("/data/player decrease distance to c4 over 20s " + columnTeam + " " + iStr).read<std::vector<bool>>();
                /*
                vector<string> deltaPosNames;
                for (size_t deltaPosIndex = 0; deltaPosIndex < delta_pos_grid_num_cells; deltaPosIndex++) {
                    string deltaPosIndexStr = std::to_string(deltaPosIndex);
                    columnData[columnPlayer].deltaPos[deltaPosIndex] =
                            file.getDataSet("/data/delta pos " + deltaPosIndexStr + " " + columnTeam + " " + iStr).read<std::vector<bool>>();
                }
                 */
                for (int radialVelindex = 0; radialVelindex < weapon_speed::num_radial_bins; radialVelindex++) {
                    columnData[columnPlayer].radialVel[radialVelindex] =
                            file.getDataSet("/data/radial vel " + std::to_string(radialVelindex) + " " + columnTeam + " " + iStr).read<std::vector<bool>>();
                }
                for (int futureTick = 0; futureTick < num_future_ticks; futureTick++) {
                    for (int radialVelindex = 0; radialVelindex < weapon_speed::num_radial_bins; radialVelindex++) {
                        columnData[columnPlayer].futureRadialVel[futureTick][radialVelindex] =
                                file.getDataSet("/data/radial vel " + std::to_string(radialVelindex) + " "
                                           + columnTeam + " " + iStr + " t+" + std::to_string(futureTick+1))
                                           .read<std::vector<bool>>();
                    }
                }
            }
        }
        size = static_cast<int64_t>(valid.size());
    }
}
