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

    struct EngagementPossibleEnemy {
        CSGOId playerId;
        EngagementEnemyState enemyState;
        double timeSinceLastVisible;
        double timeToBecomeVisible;
    };

    struct TargetPossibleEnemy {
        CSGOId playerId;
        double worldDistanceToEnemy;
        double crosshairDistanceToEnemyHead;
    };

    struct TargetPossibleEnemyLabel {
        int64_t playerId;
        bool nearestTargetEnemy;
        bool hitTargetEnemy;
    };

    class FeatureStoreResult : public QueryResult {
        vector<EngagementPossibleEnemy> engagementPossibleEnemyBuffer;
        vector<TargetPossibleEnemy> targetPossibleEnemyBuffer;
        vector<TargetPossibleEnemyLabel> targetPossibleEnemyLabelBuffer;
        bool hitEngagementBuffer;
        bool visibleEngagementBuffer;

    public:
        struct ColumnEnemyData {
            vector<int64_t> playerId;
            // inputs
            vector<EngagementEnemyState> enemyEngagementStates;
            vector<double> timeSinceLastVisible;
            vector<double> timeToBecomeVisible;
            vector<double> worldDistanceToEnemy;
            vector<double> crosshairDistanceToEnemy;
            // labels
            vector<bool> nearestTargetEnemy;
            vector<bool> hitTargetEnemy;
        };
        array<ColumnEnemyData, maxEnemies> columnEnemyData;
        vector<bool> hitEngagement;
        vector<bool> visibleEngagement;
        bool training;

        FeatureStoreResult(bool training);


        void addEngagementPossibleEnemy(const EngagementPossibleEnemy & engagementPossibleEnemy);
        void addTargetPossibleEnemy(const TargetPossibleEnemy & targetPossibleEnemy);
        void addEngagementLabel(bool hitEngagement, bool visibleEngagement);
        void addTargetPossibleEnemyLabel(const TargetPossibleEnemyLabel & targetPossibleEnemyLabel);
        void commitRow();
        void toHDF5Inner(HighFive::File & file) override;

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override { return {}; }
        void oneLineToCSV(int64_t index, std::ostream &s) override { }
        vector<string> getForeignKeyNames() override { return {}; }
        vector<string> getOtherColumnNames() override { return {}; }
    };

}

#endif //CSKNOW_FEATURE_STORE_H
