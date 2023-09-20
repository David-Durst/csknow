//
// Created by durst on 4/6/23.
//

#ifndef CSKNOW_FEATURE_STORE_PRECOMMIT_H
#define CSKNOW_FEATURE_STORE_PRECOMMIT_H

#include "queries/moments/key_retake_events.h"
#include "queries/orders.h"
#include "bots/analysis/weapon_speed.h"
#include "queries/distance_to_places.h"
#include "queries/lookback.h"
#include "queries/query.h"
#include "bots/load_save_bot_data.h"
#include "geometryNavConversions.h"
#include "queries/nearest_nav_cell.h"
#include "circular_buffer.h"
#include "weapon_id_converter.h"
#include <map>

namespace csknow::feature_store {
    using key_retake_events::max_enemies;
    constexpr double maxWorldDistance = 4000.;
    constexpr int prior_tick_spacing = 16;
    constexpr int num_prior_ticks = 24;
    constexpr int num_future_ticks = 2;

    enum class DecreaseTimingOption {
        s5,
        s10,
        s20
    };


    struct PlayerTickCounters {
        int64_t ticksSinceHurt, ticksSinceFire, ticksSinceNoFOVEnemyVisible, ticksSinceFOVEnemyVisible;
    };

    // large enough to be a long time, small enough that it won't overflow in a round
    constexpr int64_t default_many_ticks = 1e6;

    constexpr PlayerTickCounters default_tick_counters {
        default_many_ticks, default_many_ticks, default_many_ticks, default_many_ticks
    };

    struct BTTeamPlayerData {
        int64_t playerId;
        TeamId teamId;
        AreaId curArea;
        int64_t curAreaIndex;
        Vec2 curViewAngle;
        Vec3 curFootPos;
        Vec3 velocity;
        double nearestCrosshairDistanceToEnemy;
        int health, armor;
        EngineWeaponId weaponId;
        bool scoped;
        bool airborne;
        bool walking;
        bool ducking;
    };

    struct C4MapData {
        Vec3 c4Pos;
        bool c4Planted;
        int64_t ticksSincePlant;
        AreaId c4AreaId;
        int64_t c4AreaIndex;
    };

    enum class EngagementEnemyState {
        Visible,
        Communicated,
        Remembered,
        None
    };

    struct EngagementPossibleEnemy {
        CSGOId playerId;
        EngagementEnemyState enemyState;
        double timeSinceLastVisibleOrToBecomeVisible;
        double worldDistanceToEnemy;
        double crosshairDistanceToEnemyHead;
    };

    struct EngagementTeammate {
        CSGOId playerId;
        double worldDistanceToTeammate;
        double crosshairDistanceToTeammateHead;
    };

    struct TargetPossibleEnemyLabel {
        int64_t playerId;
        bool nearestTargetEnemy;
        bool hitTargetEnemy;
    };

    struct FeatureStorePreCommitBuffer {

        std::map<int64_t, int> tPlayerIdToIndex, ctPlayerIdToIndex;
        C4MapData c4MapData;

        void updateFeatureStoreBufferPlayers(const ServerState &state, bool newRound);

        vector<EngagementPossibleEnemy> engagementPossibleEnemyBuffer;
        vector<TargetPossibleEnemyLabel> targetPossibleEnemyLabelBuffer;
        bool hitEngagementBuffer;
        bool visibleEngagementBuffer;
        vector<EngagementTeammate> engagementTeammateBuffer;

        void addEngagementPossibleEnemy(const EngagementPossibleEnemy &engagementPossibleEnemy);

        void addEngagementLabel(bool hitEngagement, bool visibleEngagement);

        void addTargetPossibleEnemyLabel(const TargetPossibleEnemyLabel &targetPossibleEnemyLabel);

        void addEngagementTeammate(const EngagementTeammate &engagementTeammate);

        vector<BTTeamPlayerData> btTeamPlayerData;
        CircularBuffer<std::map<int64_t, BTTeamPlayerData>>
            historicalPlayerDataBuffer{prior_tick_spacing * num_prior_ticks + 1};
        map<CSGOId, PlayerTickCounters> playerTickCounters;

        void updatePlayerTickCounters(const ServerState & state);
        void updateCurTeamData(const ServerState & state, const nav_mesh::nav_file & navFile);
        void appendPlayerHistory();
        void clearHistory();
        int64_t getPlayerOldestContiguousHistoryIndex(int64_t playerId);
    };
}
#endif //CSKNOW_FEATURE_STORE_PRECOMMIT_H
