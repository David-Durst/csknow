//
// Created by durst on 8/3/23.
//

#ifndef CSKNOW_HUMANNESS_METRICS_H
#define CSKNOW_HUMANNESS_METRICS_H

#include "bots/analysis/feature_store_team.h"

namespace csknow::humanness_metrics {
    class HumannessMetrics : public QueryResult {
    public:
        // per key event metrics
        vector<float> velocityWhenFiring;
        vector<float> velocityWhenShot;

        vector<float> distanceToNearestTeammateWhenFiring;
        vector<float> distanceToNearestEnemyWhenFiring;
        vector<float> distanceToNearestTeammateWhenShot;
        vector<float> distanceToAttackerWhenShot;

        vector<float> distanceToCoverWhenFiring;
        vector<float> distanceToCoverWhenShot;

        // per round metrics
        vector<float> pctTimeMaxSpeedCT, pctTimeMaxSpeedT;
        vector<float> pctTimeStillCT, pctTimeStillT;
        vector<bool> ctWins;

        HumannessMetrics(const csknow::feature_store::TeamFeatureStoreResult & teamFeatureStoreResult,
                         const Games & games, const Rounds & rounds, const Players & players, const Ticks & ticks,
                         const PlayerAtTick & playerAtTick, const Hurt & hurt, const WeaponFire & weaponFire,
                         const ReachableResult & reachable, const VisPoints & visPoints);

        void toHDF5Inner(HighFive::File & file) override;

        vector<int64_t> filterByForeignKey(int64_t) override { return {}; }
        void oneLineToCSV(int64_t, std::ostream &) override { }
        vector<string> getForeignKeyNames() override { return {}; }
        vector<string> getOtherColumnNames() override { return {}; }
    };


}

#endif //CSKNOW_HUMANNESS_METRICS_H
