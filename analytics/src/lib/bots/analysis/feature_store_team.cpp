//
// Created by durst on 4/6/23.
//

#include "bots/analysis/feature_store_team.h"

namespace csknow::feature_store {
    void TeamFeatureStoreResult::init(size_t size) {
        roundId.resize(size, INVALID_ID);
        tickId.resize(size, INVALID_ID);
        c4Status.resize(size, C4Status::NotPlanted);
        for (int i = 0; i < maxEnemies; i++) {
            columnTData[i].playerId.resize(size, INVALID_ID);
            columnTData[i].distanceToASite.resize(size, INVALID_ID);
            columnTData[i].distanceToBSite.resize(size, INVALID_ID);
            columnCTData[i].playerId.resize(size, INVALID_ID);
            columnCTData[i].distanceToASite.resize(size, INVALID_ID);
            columnCTData[i].distanceToBSite.resize(size, INVALID_ID);
            for (int j = 0; j < num_orders_per_site; j++) {
                columnTData[i].distanceToNearestAOrderNavArea[j].resize(size, INVALID_ID);
                columnTData[i].distanceToNearestBOrderNavArea[j].resize(size, INVALID_ID);
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
    }

    TeamFeatureStoreResult::TeamFeatureStoreResult(size_t size, const std::vector<csknow::orders::QueryOrder> & orders) {
        init(size);
        for (const auto & order : orders) {
            if (order.orderType == orders::OrderType::AOrder) {
                aOrders.push_back(order);
            }
            else {
                bOrders.push_back(order);
            }
        }
    }

    void TeamFeatureStoreResult::commitTeamRow(FeatureStorePreCommitBuffer & buffer,
                                               DistanceToPlacesResult & distanceToPlaces,
                                               const nav_mesh::nav_file & navFile,
                                               int64_t roundIndex, int64_t tickIndex) {
        roundId[tickIndex] = roundIndex;
        tickId[tickIndex] = tickIndex;
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
        for (size_t i = 0; i < buffer.btTeamPlayerData.size(); i++) {
            const BTTeamPlayerData & btTeamPlayerData = buffer.btTeamPlayerData[i];
            auto & columnData = btTeamPlayerData.teamId == ENGINE_TEAM_T ? columnTData : columnCTData;
            size_t columnIndex = btTeamPlayerData.teamId == ENGINE_TEAM_T ?
                buffer.tPlayerIdToIndex[btTeamPlayerData.playerId] : buffer.ctPlayerIdToIndex[btTeamPlayerData.playerId];

            columnData[columnIndex].playerId[tickIndex] = btTeamPlayerData.playerId;
            columnData[columnIndex].distanceToASite[tickIndex] =
                distanceToPlaces.getClosestDistance(btTeamPlayerData.curArea, a_site, navFile);
            columnData[columnIndex].distanceToBSite[tickIndex] =
                distanceToPlaces.getClosestDistance(btTeamPlayerData.curArea, b_site, navFile);
            for (size_t j = 0; j < num_orders_per_site; j++) {
                double & aOrderDistance = columnData[columnIndex].distanceToNearestAOrderNavArea[j][tickIndex];
                aOrderDistance = std::numeric_limits<double>::max();
                for (size_t k = 0; k < aOrders[j].places.size(); k++) {
                    aOrderDistance = distanceToPlaces.getMedianDistance(btTeamPlayerData.curAreaIndex, aOrders[j].places[k]);
                }
                double & bOrderDistance = columnData[columnIndex].distanceToNearestBOrderNavArea[j][tickIndex];
                bOrderDistance = std::numeric_limits<double>::max();
                for (size_t k = 0; k < bOrders[j].places.size(); k++) {
                    bOrderDistance = distanceToPlaces.getMedianDistance(btTeamPlayerData.curAreaIndex, bOrders[j].places[k]);
                }
            }
        }
    }

    void computeAcausalLabels(const Games & games, const Rounds & rounds,
                              const Ticks & ticks) {

    }

}
