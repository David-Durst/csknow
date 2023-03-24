//
// Created by durst on 3/3/23.
//

#ifndef CSKNOW_FEATURE_STORE_H
#define CSKNOW_FEATURE_STORE_H

#include "queries/query.h"
#include "bots/load_save_bot_data.h"


namespace csknow::feature_store {
    constexpr int maxEnemies = 5;
    enum class EngagementEnemyState {
        Visible,
        Communicated,
        Remembered,
        None
    };

    constexpr double maxTimeToVis = 100.;
    constexpr double maxWorldDistance = 4000.;
    constexpr double maxCrosshairDistance = 30.;
    constexpr double maxPositionDelta = 150.;
    constexpr double maxViewAngleDelta = 15.;
    constexpr size_t windowSize = 10;
    struct EngagementPossibleEnemy {
        CSGOId playerId;
        EngagementEnemyState enemyState;
        double timeSinceLastVisibleOrToBecomeVisible;
        double worldDistanceToEnemy;
        double crosshairDistanceToEnemyHead;
    };

    struct TargetPossibleEnemyLabel {
        int64_t playerId;
        bool nearestTargetEnemy;
        bool hitTargetEnemy;
    };

    struct FeatureStorePreCommitBuffer {
        std::map<int64_t, int> tPlayerIdToIndex, ctPlayerIdToIndex;
        void updateFeatureStoreBufferPlayers(const ServerState & state);

        vector<EngagementPossibleEnemy> engagementPossibleEnemyBuffer;
        vector<TargetPossibleEnemyLabel> targetPossibleEnemyLabelBuffer;
        bool hitEngagementBuffer;
        bool visibleEngagementBuffer;

        void addEngagementPossibleEnemy(const EngagementPossibleEnemy & engagementPossibleEnemy);
        void addEngagementLabel(bool hitEngagement, bool visibleEngagement);
        void addTargetPossibleEnemyLabel(const TargetPossibleEnemyLabel & targetPossibleEnemyLabel);
    };

    class FeatureStoreResult : public QueryResult {
        void init(size_t size);

    public:
        vector<int64_t> roundId;
        vector<int64_t> tickId;
        vector<int64_t> playerId;
        vector<int64_t> patId;
        struct ColumnEnemyData {
            vector<int64_t> playerId;
            // inputs
            vector<EngagementEnemyState> enemyEngagementStates;
            vector<double> timeSinceLastVisibleOrToBecomeVisible;
            vector<double> worldDistanceToEnemy;
            vector<double> crosshairDistanceToEnemy;
            // labels
            vector<bool> nearestTargetEnemy;
            vector<bool> hitTargetEnemy;
            vector<bool> visibleIn1s, visibleIn2s, visibleIn5s, visibleIn10s;
        };
        array<ColumnEnemyData, maxEnemies> columnEnemyData;
        vector<bool> hitEngagement;
        vector<bool> visibleEngagement;
        vector<int> nearestCrosshairCurTick, nearestCrosshairEnemy500ms, nearestCrosshairEnemy1s, nearestCrosshairEnemy2s;
        vector<double> positionOffset2sUpToThreshold, viewAngleOffset2sUpToThreshold;
        // these are just used to create binomial distributions
        vector<double> negPositionOffset2sUpToThreshold, negViewAngleOffset2sUpToThreshold;
        array<vector<double>, maxEnemies+1> pctNearestCrosshairEnemy2s;
        vector<int64_t> nextPATId2s;
        vector<bool> valid;
        bool training;

        // for use in non-multithreaded applications where want on buffer
        FeatureStorePreCommitBuffer defaultBuffer;

        FeatureStoreResult();
        FeatureStoreResult(size_t size);

        void commitRow(FeatureStorePreCommitBuffer & buffer, size_t rowIndex = 0,
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
