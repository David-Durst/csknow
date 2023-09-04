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
        vector<float> unscaledSpeed;
        vector<float> unscaledSpeedWhenFiring;
        vector<float> unscaledSpeedWhenShot;

        vector<float> scaledSpeed;
        vector<float> scaledSpeedWhenFiring;
        vector<float> scaledSpeedWhenShot;

        vector<float> weaponOnlyScaledSpeed;
        vector<float> weaponOnlyScaledSpeedWhenFiring;
        vector<float> weaponOnlyScaledSpeedWhenShot;

        vector<float> distanceToNearestTeammate;
        vector<float> distanceToNearestTeammateWhenFiring;
        vector<float> distanceToNearestTeammateWhenShot;

        vector<float> deltaDistanceToNearestTeammate;
        vector<float> deltaDistanceToNearestTeammateWhenFiring;
        vector<float> deltaDistanceToNearestTeammateWhenShot;

        vector<float> distanceToNearestEnemy;
        vector<float> distanceToNearestEnemyWhenFiring;
        vector<float> distanceToNearestEnemyWhenShot;

        vector<float> distanceToAttackerWhenShot;

        vector<float> distanceToCover;
        vector<float> distanceToCoverWhenEnemyVisibleNoFOV;
        vector<float> distanceToCoverWhenEnemyVisibleFOV;
        vector<float> distanceToCoverWhenFiring;
        vector<float> distanceToCoverWhenShot;

        vector<float> distanceToC4;
        vector<float> distanceToC4WhenEnemyVisibleFOV;
        vector<float> distanceToC4WhenFiring;
        vector<float> distanceToC4WhenShot;

        vector<float> deltaDistanceToC4;
        vector<float> deltaDistanceToC4WhenEnemyVisibleFOV;
        vector<float> deltaDistanceToC4WhenFiring;
        vector<float> deltaDistanceToC4WhenShot;

        vector<float> timeFromFiringToTeammateSeeingEnemyFOV;
        vector<float> timeFromShotToTeammateSeeingEnemyFOV;

        // per round metrics
        vector<float> pctTimeMaxSpeedCT, pctTimeMaxSpeedT;
        vector<float> pctTimeStillCT, pctTimeStillT;
        vector<bool> ctWins;

        // round ids for filtering to push rounds
        vector<int64_t> roundIdPerPAT, roundIdPerFiringPAT, roundIdPerShotPAT,
            roundIdPerNearestTeammate, roundIdPerNearestTeammateFiring, roundIdPerNearestTeammateShot,
            roundIdPerEnemyVisibleNoFOVPAT, roundIdPerEnemyVisibleFOVPAT,
            roundIdPerFiringToTeammateSeeingEnemy, roundIdPerShotToTeammateSeeingEnemy;
        vector<bool> isCTPerPAT, isCTPerFiringPAT, isCTPerShotPAT,
                isCTPerNearestTeammate, isCTPerNearestTeammateFiring, isCTPerNearestTeammateShot,
                isCTPerEnemyVisibleNoFOVPAT, isCTPerEnemyVisibleFOVPAT,
                isCTPerFiringToTeammateSeeingEnemy, isCTPerShotToTeammateSeeingEnemy;
        vector<int64_t> playerIdPerPAT, playerIdPerFiringPAT, playerIdPerShotPAT,
                playerIdPerNearestTeammate, playerIdPerNearestTeammateFiring, playerIdPerNearestTeammateShot,
                playerIdPerEnemyVisibleNoFOVPAT, playerIdPerEnemyVisibleFOVPAT,
                playerIdPerFiringToTeammateSeeingEnemy, playerIdPerShotToTeammateSeeingEnemy;
        vector<int64_t> roundIdPerRound;


        HumannessMetrics(const csknow::feature_store::TeamFeatureStoreResult & teamFeatureStoreResult,
                         const Games & games, const Rounds & rounds, const Players & players, const Ticks & ticks,
                         const PlayerAtTick & playerAtTick, const Hurt & hurt, const WeaponFire & weaponFire,
                         const ReachableResult & reachable, const VisPoints & visPoints, bool enable);

        void toHDF5Inner(HighFive::File & file) override;

        vector<int64_t> filterByForeignKey(int64_t) override { return {}; }
        void oneLineToCSV(int64_t, std::ostream &) override { }
        vector<string> getForeignKeyNames() override { return {}; }
        vector<string> getOtherColumnNames() override { return {}; }
    };


}

#endif //CSKNOW_HUMANNESS_METRICS_H
