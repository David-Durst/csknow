//
// Created by durst on 12/25/22.
//

#ifndef CSKNOW_STREAMING_MANAGER_H
#define CSKNOW_STREAMING_MANAGER_H

#include <bots/streaming_bot_database.h>
#include <bots/streaming_moments/streaming_fire_history.h>
#include <bots/streaming_moments/streaming_engagement_aim.h>
#include <bots/streaming_moments/streaming_test_logger.h>
#include "queries/lookback.h"

// when converting col store to row store, precompute plant/defusal for entire round
struct RoundPlantDefusal {
    int64_t plantTickIndex;
    int64_t defusalTickIndex;
};
RoundPlantDefusal processRoundPlantDefusals(const Rounds & rounds, const Ticks & ticks, const Plants & plants,
                                            const Defusals & defusals, int64_t roundIndex);

class StreamingManager {
public:
    StreamingBotDatabase db;
    csknow::test_log::StreamingTestLogger streamingTestLogger;
    csknow::fire_history::StreamingFireHistory streamingFireHistory;
    csknow::engagement_aim::StreamingEngagementAim streamingEngagementAim;
    bool forceReset = false;

    StreamingManager(const string & navPath) : streamingTestLogger(navPath), streamingEngagementAim(navPath)  { }
    void update(const ServerState & state);

    void update(const Games & games, const RoundPlantDefusal & roundPlantDefusal, const Rounds & rounds,
                const Players & players, const Ticks & ticks, const WeaponFire & weaponFire, const Hurt & hurt,
                const PlayerAtTick & playerAtTick, int64_t tickIndex,
                const csknow::nearest_nav_cell::NearestNavCell & nearestNavCell, const VisPoints & visPoints,
                const TickRates & tickRates, bool computeVisibility = true);
};

bool demoIsVisible(const PlayerAtTick & playerAtTick, int64_t attackerPATId, int64_t victimPATId,
                   const csknow::nearest_nav_cell::NearestNavCell & nearestNavCell,
                   const VisPoints & visPoints);
bool vecIsVisible(Vec3 attackerEyePos, Vec3 victimEyePos, Vec2 curViewAngle,
                  const csknow::nearest_nav_cell::NearestNavCell & nearestNavCell,
                  const VisPoints & visPoints);

#endif //CSKNOW_STREAMING_MANAGER_H
