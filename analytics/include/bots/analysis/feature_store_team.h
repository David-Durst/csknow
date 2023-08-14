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
#include "bots/analysis/weapon_speed.h"
#include "queries/orders.h"
#include "circular_buffer.h"
#include "queries/moments/key_retake_events.h"

namespace csknow::feature_store {
    constexpr double invalidWorldPosDim = 8000.;
    const Vec3 invalidWorldPos = {invalidWorldPosDim, invalidWorldPosDim, invalidWorldPosDim};
    constexpr int num_orders_per_site = 3;
    constexpr int num_places = 26;
    constexpr int area_grid_dim = 5;
    constexpr int area_grid_size = area_grid_dim*area_grid_dim;
    constexpr int delta_pos_grid_radius = 130;
    constexpr int delta_pos_grid_cell_dim = 20;
    constexpr int delta_pos_z_num_cells = 3;
    constexpr int delta_pos_grid_num_cells = delta_pos_z_num_cells *
            (delta_pos_grid_radius * 2 * delta_pos_grid_radius * 2) /
            (delta_pos_grid_cell_dim * delta_pos_grid_cell_dim);
    const int delta_pos_grid_num_cells_per_xy_dim = static_cast<int>(std::sqrt(delta_pos_grid_num_cells / delta_pos_z_num_cells));
    const int delta_pos_grid_num_xy_cells_per_z_change = delta_pos_grid_num_cells_per_xy_dim * delta_pos_grid_num_cells_per_xy_dim;
    constexpr double seconds_per_c4_timer_bucket = 10.;
    constexpr int num_c4_timer_buckets = 4;
    constexpr float c4_max_time_seconds = 40.;
    constexpr int every_nth_row = 10;
    const string a_site = "BombsiteA", b_site = "BombsiteB";
    const Vec3 zeroVec = {0., 0., 0.};
    const Vec2 zeroVec2D = {0., 0.};
    constexpr float crosshair_max_distance = 30.f;

    enum class C4Status {
        PlantedA,
        PlantedB,
        NotPlanted
    };

    class TeamFeatureStoreResult : public QueryResult {
        void init(size_t size);

    public:
        std::vector<csknow::orders::QueryOrder> aOrders, bOrders;
        bool curBaiting = false;

        // the following are tracked for all ticks so humanness metrics can use them
        struct NonDecimatedPlayerData {
            vector<int64_t> playerId;
            vector<int64_t> areaIndex;
            vector<int64_t> areaId;
            vector<bool> noFOVEnemyVisible;
            vector<bool> fovEnemyVisible;
        };
        array<NonDecimatedPlayerData, max_enemies> nonDecimatedCTData, nonDecimatedTData;

        vector<int64_t> tickIdToInternalId;
        vector<int64_t> internalIdToTickId;
        // tracks valid ticks (even if not used for training due to decimation)
        vector<bool> nonDecimatedValidRetakeTicks;
        // demo file names are only stored once per demo, not for all ticks, use gameId to lookup into them
        vector<string> demoFile;
        vector<int64_t> gameId;
        vector<int64_t> roundId;
        vector<int64_t> roundNumber;
        vector<int64_t> tickId;
        vector<int64_t> gameTickNumber;
        vector<bool> valid;
        vector<bool> freezeTimeEnded;
        vector<bool> retakeSaveRoundTick;
        vector<string> testName;
        vector<bool> testSuccess;
        vector<bool> baiting;

        vector<int64_t> c4AreaIndex;
        vector<C4Status> c4Status;
        vector<bool> c4PlantA;
        vector<bool> c4PlantB;
        vector<bool> c4NotPlanted;
        vector<int64_t> c4TicksSincePlant;
        vector<float> c4TimeLeftPercent;
        array<vector<bool>, num_c4_timer_buckets> c4TimerBucketed;
        vector<Vec3> c4Pos;
        vector<float> c4DistanceToASite, c4DistanceToBSite;
        //array<vector<float>, num_orders_per_site> c4DistanceToNearestAOrderNavArea, c4DistanceToNearestBOrderNavArea;

        struct ColumnPlayerData {
            vector<int64_t> playerId;
            // inputs
            array<vector<bool>, max_enemies> indexOnTeam;
            vector<bool> ctTeam;
            vector<bool> alive;
            vector<Vec2> viewAngle;
            vector<Vec3> footPos;
            vector<Vec3> velocity;
            //vector<Vec3> alignedFootPos;
            array<vector<Vec3>, num_prior_ticks> priorVelocity;
            array<vector<Vec3>, num_prior_ticks> priorFootPos;
            array<vector<bool>, num_prior_ticks> priorFootPosValid;
            // state inputs
            vector<float> nearestCrosshairDistanceToEnemy;
            array<vector<float>, num_prior_ticks> priorNearestCrosshairDistanceToEnemy;
            // 0 means not shot or visible, 1 means shot cur frame or enemy currently visible
            vector<float> hurtInLast5s, fireInLast5s, noFOVEnemyVisibleInLast5s, fovEnemyVisibleInLast5s;
            vector<float> health, armor;
            // control inputs
            vector<int64_t> areaIndex;
            vector<bool> decreaseDistanceToC4Over5s, decreaseDistanceToC4Over10s, decreaseDistanceToC4Over20s;
            //vector<float> distanceToASite, distanceToBSite;
            //array<vector<float>, num_orders_per_site> distanceToNearestAOrderNavArea, distanceToNearestBOrderNavArea;
            //array<vector<bool>, num_places> curPlace;
            //array<vector<bool>, area_grid_size> areaGridCellInPlace;
            // outputs
            //array<vector<float>, num_orders_per_site> distributionNearestAOrders, distributionNearestBOrders;
            //array<vector<float>, num_places> distributionNearestPlace;
            //array<vector<float>, area_grid_size> distributionNearestAreaGridInPlace;
            //array<vector<bool>, delta_pos_grid_num_cells> deltaPos;
            vector<EngineWeaponId> weaponId;
            vector<bool> scoped;
            vector<bool> airborne;
            vector<bool> walking;
            vector<bool> ducking;
            array<vector<bool>, weapon_speed::num_radial_bins> radialVel;
            array<array<vector<bool>, weapon_speed::num_radial_bins>, num_future_ticks> futureRadialVel;
        };
        array<ColumnPlayerData, max_enemies> columnCTData, columnTData;
        vector<std::reference_wrapper<const array<ColumnPlayerData, max_enemies>>> getAllColumnData() const {
            return {columnCTData, columnTData};
        }
        vector<std::reference_wrapper<array<ColumnPlayerData, max_enemies>>> getAllColumnData() {
            return {columnCTData, columnTData};
        }
        vector<string> allColumnDataTeam = {"CT", "T"};


        void setOrders(const std::vector<csknow::orders::QueryOrder> & orders);
        TeamFeatureStoreResult();
        TeamFeatureStoreResult(size_t size, const std::vector<csknow::orders::QueryOrder> & orders,
                               std::optional<std::reference_wrapper<const Ticks>> ticks = std::nullopt,
                               std::optional<std::reference_wrapper<const csknow::key_retake_events::KeyRetakeEvents>> keyRetakeEvents = std::nullopt,
                               bool requireBothTeamsAlive = false);
        virtual ~TeamFeatureStoreResult() = default;
        void reinit();
        bool commitTeamRow(const ServerState & state, FeatureStorePreCommitBuffer & buffer,
                           const DistanceToPlacesResult & distanceToPlaces,
                           const nav_mesh::nav_file & navFile,
                           int64_t roundIndex = 0, int64_t tickIndex = 0);
        enum class ACausalTimingOption {
            s6,
            s15,
            s30
        };
        /* void computeOrderACausalLabels(int64_t curTick, CircularBuffer<int64_t> & futureTracker,
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
                                          */
        void computeDeltaPosACausalLabels(int64_t curTick, CircularBuffer<int64_t> & futureTracker,
                                          array<ColumnPlayerData,max_enemies> & columnData);
        void computeFutureDeltaPosACausalLabels(int64_t curTick, CircularBuffer<int64_t> & futureTracker0_5,
                                                CircularBuffer<int64_t> & futureTracker1_0,
                                                array<ColumnPlayerData,max_enemies> & columnData,
                                                const TickRates & ticksRates);
        void removePartialACausalLabels(int64_t curTick, array<ColumnPlayerData,max_enemies> & columnData);
        void computeDecreaseDistanceToC4(int64_t curTick, CircularBuffer<int64_t> & futureTracker,
                                          array<ColumnPlayerData,max_enemies> & columnData,
                                          DecreaseTimingOption decreaseTimingOption,
                                          const ReachableResult & reachableResult);
        void computeAcausalLabels(const Games & games, const Rounds & rounds, const Ticks & ticks,
                                  const Players & players, const DistanceToPlacesResult & distanceToPlacesResult,
                                  const ReachableResult & reachableResult,
                                  const nav_mesh::nav_file & navFile,
                                  const csknow::key_retake_events::KeyRetakeEvents & keyRetakeEvents);
        void toHDF5Inner(HighFive::File & file) override;
        string fileName;
        void load(const std::string &filePath);
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
