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

namespace csknow::feature_store {
    constexpr int maxEnemies = 5;
    constexpr int num_orders_per_site = 3;
    const string a_site = "BombsiteA", b_site = "BombsiteB";

    enum class C4Status {
        PlantedA,
        PlantedB,
        NotPlanted
    };

    class TeamFeatureStoreResult : public QueryResult {
        void init(size_t size);

    public:
        vector<int64_t> roundId;
        vector<int64_t> tickId;
        vector<C4Status> c4Status;
        std::vector<csknow::orders::QueryOrder> aOrders, bOrders;

        struct ColumnPlayerData {
            vector<int64_t> playerId;
            // inputs
            vector<double> distanceToASite, distanceToBSite;
            array<vector<double>, num_orders_per_site> distanceToNearestAOrderNavArea, distanceToNearestBOrderNavArea;
            // outputs
            array<vector<double>, num_orders_per_site> distributionNearestAOrders15s, distributionNearestBOrders15s;
            array<vector<double>, num_orders_per_site> distributionNearestAOrders30s, distributionNearestBOrders30s;
        };
        array<ColumnPlayerData, maxEnemies> columnCTData, columnTData;

        TeamFeatureStoreResult(size_t size, const std::vector<csknow::orders::QueryOrder> & orders);
        void commitTeamRow(FeatureStorePreCommitBuffer & buffer, DistanceToPlacesResult & distanceToPlaces,
                           const nav_mesh::nav_file & navFile,
                           int64_t roundIndex = 0, int64_t tickIndex = 0);
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
