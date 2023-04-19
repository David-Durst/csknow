//
// Created by durst on 4/6/23.
//

#ifndef CSKNOW_FEATURE_STORE_TEAM_H
#define CSKNOW_FEATURE_STORE_TEAM_H

#include "queries/query.h"
#include "bots/load_save_bot_data.h"
#include "geometryNavConversions.h"
#include "queries/distance_to_places.h"
#include "bots/analysis/feature_store_precommit.h"
#include "queries/orders.h"
#include "circular_buffer.h"

namespace csknow::feature_store {
    constexpr double invalidWorldPosDim = 8000.;
    const Vec3 invalidWorldPos = {invalidWorldPosDim, invalidWorldPosDim, invalidWorldPosDim};
    constexpr int maxEnemies = 5;
    constexpr int num_orders_per_site = 3;
    constexpr int num_places = 27;
    constexpr int area_grid_dim = 5;
    constexpr int area_grid_size = area_grid_dim*area_grid_dim;
    const string a_site = "BombsiteA", b_site = "BombsiteB";

    enum class C4Status {
        PlantedA,
        PlantedB,
        NotPlanted
    };

    class TeamFeatureStoreResult : public QueryResult {
        void init(size_t size);

    public:
        std::vector<csknow::orders::QueryOrder> aOrders, bOrders;

        vector<int64_t> roundId;
        vector<int64_t> tickId;
        vector<bool> valid;

        vector<C4Status> c4Status;
        vector<Vec3> c4Pos;
        vector<double> c4DistanceToASite, c4DistanceToBSite;
        array<vector<double>, num_orders_per_site> c4DistanceToNearestAOrderNavArea, c4DistanceToNearestBOrderNavArea;

        struct ColumnPlayerData {
            vector<int64_t> playerId;
            // inputs
            vector<Vec3> footPos;
            vector<double> distanceToASite, distanceToBSite;
            array<vector<double>, num_orders_per_site> distanceToNearestAOrderNavArea, distanceToNearestBOrderNavArea;
            array<vector<bool>, num_places> curPlace;
            array<vector<bool>, area_grid_size> areaGridCellInPlace;
            // outputs
            array<vector<double>, num_orders_per_site> distributionNearestAOrders15s, distributionNearestBOrders15s;
            array<vector<double>, num_orders_per_site> distributionNearestAOrders30s, distributionNearestBOrders30s;
            array<vector<double>, num_places> distributionNearestPlace10to15s;
            array<vector<double>, area_grid_size> distributionNearestAreaGridInPlace10to15s;
        };
        array<ColumnPlayerData, maxEnemies> columnCTData, columnTData;
        vector<std::reference_wrapper<const array<ColumnPlayerData, maxEnemies>>> getAllColumnData() const {
            return {columnCTData, columnTData};
        }
        vector<string> allColumnDataTeam = {"CT", "T"};


        void setOrders(const std::vector<csknow::orders::QueryOrder> & orders);
        TeamFeatureStoreResult(size_t size, const std::vector<csknow::orders::QueryOrder> & orders);
        void reinit();
        void commitTeamRow(FeatureStorePreCommitBuffer & buffer, DistanceToPlacesResult & distanceToPlaces,
                           const nav_mesh::nav_file & navFile,
                           int64_t roundIndex = 0, int64_t tickIndex = 0);
        void computeTeamTickACausalLabels(int64_t curTick, CircularBuffer<int64_t> & futureTracker, array<ColumnPlayerData,
                                          maxEnemies> & columnData, bool future15s);
        void computeAcausalLabels(const Games & games, const Rounds & rounds,
                                  const Ticks & ticks);
        void toHDF5Inner(HighFive::File & file) override;

        vector<int64_t> filterByForeignKey(int64_t) override { return {}; }
        void oneLineToCSV(int64_t, std::ostream &) override { }
        vector<string> getForeignKeyNames() override { return {}; }
        vector<string> getOtherColumnNames() override { return {}; }
    };

}
#endif //CSKNOW_FEATURE_STORE_TEAM_H
