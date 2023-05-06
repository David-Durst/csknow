//
// Created by durst on 4/6/23.
//

#ifndef CSKNOW_FEATURE_STORE_TEAM_H
#define CSKNOW_FEATURE_STORE_TEAM_H

#include "queries/query.h"
#include "queries/lookback.h"
#include "bots/load_save_bot_data.h"
#include "geometryNavConversions.h"
#include "queries/distance_to_places.h"
#include "bots/analysis/feature_store_precommit.h"
#include "queries/orders.h"
#include "circular_buffer.h"
#include "queries/moments/key_retake_events.h"

namespace csknow::feature_store {
    constexpr double invalidWorldPosDim = 8000.;
    const Vec3 invalidWorldPos = {invalidWorldPosDim, invalidWorldPosDim, invalidWorldPosDim};
    constexpr int maxEnemies = 5;
    constexpr int num_orders_per_site = 3;
    constexpr int num_places = 26;
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
        vector<bool> retakeSaveRoundTick;

        vector<C4Status> c4Status;
        vector<int64_t> c4TicksSincePlant;
        vector<Vec3> c4Pos;
        vector<float> c4DistanceToASite, c4DistanceToBSite;
        array<vector<float>, num_orders_per_site> c4DistanceToNearestAOrderNavArea, c4DistanceToNearestBOrderNavArea;

        struct ColumnPlayerData {
            vector<int64_t> playerId;
            // inputs
            array<vector<bool>, maxEnemies> indexOnTeam;
            vector<bool> ctTeam;
            vector<Vec3> footPos;
            array<vector<Vec3>, num_prior_ticks> priorFootPos;
            vector<Vec3> velocity;
            vector<float> distanceToASite, distanceToBSite;
            array<vector<float>, num_orders_per_site> distanceToNearestAOrderNavArea, distanceToNearestBOrderNavArea;
            array<vector<bool>, num_places> curPlace;
            array<array<vector<bool>, num_places>, num_prior_ticks> priorPlaces;
            array<vector<bool>, area_grid_size> areaGridCellInPlace;
            array<array<vector<bool>, area_grid_size>, num_prior_ticks> priorAreaGridCellInPlace;
            // outputs
            array<vector<float>, num_orders_per_site> distributionNearestAOrders, distributionNearestBOrders;
            //array<vector<float>, num_orders_per_site> distributionNearestAOrders15s, distributionNearestBOrders15s;
            //array<vector<float>, num_orders_per_site> distributionNearestAOrders30s, distributionNearestBOrders30s;
            array<vector<float>, num_places> distributionNearestPlace;
            //array<vector<float>, num_places> distributionNearestPlace7to15s;
            array<vector<float>, area_grid_size> distributionNearestAreaGridInPlace;
            //array<vector<float>, area_grid_size> distributionNearestAreaGridInPlace7to15s;
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
        enum class ACausalTimingOption {
            s6,
            s15,
            s30
        };
        void computeOrderACausalLabels(int64_t curTick, CircularBuffer<int64_t> & futureTracker,
                                       array<ColumnPlayerData, maxEnemies> & columnData, ACausalTimingOption timingOption);
        void computePlaceACausalLabels(const Games & games, const Ticks & ticks, const TickRates & tickRates,
                                       int64_t curGame, int64_t curTick,
                                       CircularBuffer<int64_t> & futureTracker,
                                       array<ColumnPlayerData,maxEnemies> & columnData,
                                       double futureSecondsTheshold, const Players & players,
                                       const DistanceToPlacesResult & distanceToPlacesResult,
                                       const nav_mesh::nav_file & navFile);
        void computeAreaACausalLabels(const Ticks & ticks, const TickRates & tickRates, int64_t curTick,
                                           CircularBuffer<int64_t> & futureTracker,
                                           array<ColumnPlayerData,maxEnemies> & columnData,
                                           double futureSecondsTheshold);
        void computeAcausalLabels(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                  const Players & players, const DistanceToPlacesResult & distanceToPlacesResult,
                                  const nav_mesh::nav_file & navFile,
                                  const csknow::key_retake_events::KeyRetakeEvents & keyRetakeEvents);
        void toHDF5Inner(HighFive::File & file) override;
        /*
        void checkPossiblyBadValue() {
            std::cout << "checking possibly bad value on init " << columnTData[4].distributionNearestAOrders15s[0][8240] << std::endl;
        }
         */

        vector<int64_t> filterByForeignKey(int64_t) override { return {}; }
        void oneLineToCSV(int64_t, std::ostream &) override { }
        vector<string> getForeignKeyNames() override { return {}; }
        vector<string> getOtherColumnNames() override { return {}; }
    };

}
#endif //CSKNOW_FEATURE_STORE_TEAM_H
