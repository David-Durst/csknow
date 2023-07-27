//
// Created by durst on 3/3/23.
//

#ifndef CSKNOW_FEATURE_STORE_H
#define CSKNOW_FEATURE_STORE_H

#include "queries/query.h"
#include "bots/load_save_bot_data.h"
#include "bots/analysis/feature_store_team.h"
#include "feature_store_precommit.h"


namespace csknow::feature_store {
    enum class NearestEnemyState {
        Decrease,
        Constant,
        Increase,
        NUM_NEAREST_ENEMY_STATE
    };
    constexpr size_t numNearestEnemyState = static_cast<size_t>(NearestEnemyState::NUM_NEAREST_ENEMY_STATE);

    constexpr double maxTimeToVis = 100.;
    constexpr double maxCrosshairDistance = 30.;
    constexpr double maxTeammateCrosshairDistance = 180.;
    constexpr double maxPositionDelta = 150.;
    constexpr double maxViewAngleDelta = 15.;
    constexpr size_t windowSize = 10;
    constexpr double nearestEnemyChangeThreshold = 1.;

    class FeatureStoreResult : public QueryResult {
        void init(size_t size);

    public:
        bool disable = true;
        vector<int64_t> roundId;
        vector<int64_t> tickId;
        vector<int64_t> playerId;
        vector<int64_t> patId;
        struct ColumnEnemyData {
            vector<int64_t> playerId;
            // target
            // inputs
            vector<EngagementEnemyState> enemyEngagementStates;
            vector<float> timeSinceLastVisibleOrToBecomeVisible;
            vector<float> worldDistanceToEnemy;
            vector<float> crosshairDistanceToEnemy;
            // labels
            vector<bool> nearestTargetEnemy;
            vector<bool> hitTargetEnemy;
            vector<bool> visibleIn1s, visibleIn2s, visibleIn5s, visibleIn10s;
        };
        array<ColumnEnemyData, max_enemies> columnEnemyData;
        struct ColumnTeammateData {
            vector<int64_t> playerId;
            // inputs
            vector<float> teammateWorldDistance;
            vector<float> crosshairDistanceToTeammate;
        };
        array<ColumnTeammateData, max_enemies> columnTeammateData;
        vector<bool> fireCurTick;
        vector<bool> hitEngagement;
        vector<bool> visibleEngagement;
        vector<int> nearestCrosshairCurTick, nearestCrosshairEnemy500ms, nearestCrosshairEnemy1s, nearestCrosshairEnemy2s;
        vector<float> positionOffset2sUpToThreshold, viewAngleOffset2sUpToThreshold;
        // these are just used to create binomial distributions
        vector<float> negPositionOffset2sUpToThreshold, negViewAngleOffset2sUpToThreshold;
        array<vector<float>, max_enemies + 1> pctNearestCrosshairEnemy2s;
        vector<float> visibleEnemy2s, negVisibleEnemy2s, fireNext2s, negFireNext2s;
        array<vector<float>, numNearestEnemyState> pctNearestEnemyChange2s;
        vector<int64_t> nextPATId2s;
        vector<bool> valid;
        bool training;

        // for use in non-multithreaded applications where want on buffer
        FeatureStorePreCommitBuffer defaultBuffer;

        // team data, only one row per tick so different data layout
        TeamFeatureStoreResult teamFeatureStoreResult;

        FeatureStoreResult();
        FeatureStoreResult(const Ticks & ticks, size_t tickSize, size_t patSize,
                           const std::vector<csknow::orders::QueryOrder> & orders,
                           const csknow::key_retake_events::KeyRetakeEvents & keyRetakeEvents);
        void reinit();

        void commitPlayerRow(FeatureStorePreCommitBuffer & buffer, size_t rowIndex = 0,
                             int64_t roundIndex = 0, int64_t tickIndex = 0, int64_t playerIndex = 0);
        void computeAcausalLabels(const Games & games, const Rounds & rounds,
                                  const Ticks & ticks, const PlayerAtTick & playerAtTick);
        FeatureStoreResult makeWindows() const;
        void toHDF5Inner(HighFive::File & file) override;

        vector<int64_t> filterByForeignKey(int64_t) override { return {}; }
        void oneLineToCSV(int64_t, std::ostream &) override { }
        vector<string> getForeignKeyNames() override { return {}; }
        vector<string> getOtherColumnNames() override { return {}; }
        void checkInvalid() const;
    };

}

#endif //CSKNOW_FEATURE_STORE_H
