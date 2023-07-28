//
// Created by durst on 4/6/23.
//

#ifndef CSKNOW_FEATURE_STORE_PRECOMMIT_H
#define CSKNOW_FEATURE_STORE_PRECOMMIT_H

#include "queries/moments/key_retake_events.h"
#include "queries/orders.h"
#include "bots/analysis/weapon_speed.h"
#include "bots/analysis/feature_store_precommit.h"
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
    constexpr double maxWorldDistance = 4000.;
    constexpr int prior_tick_spacing = 64;
    constexpr int num_prior_ticks = 12;
    constexpr int num_future_ticks = 2;
    constexpr int max_enemies = 5;

    enum class DecreaseTimingOption {
        s5,
        s10,
        s20
    };


    struct BTTeamPlayerData {
        int64_t playerId;
        TeamId teamId;
        AreaId curArea;
        int64_t curAreaIndex;
        Vec3 curFootPos;
        Vec3 velocity;
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

        void updateFeatureStoreBufferPlayers(const ServerState &state);

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

        void updateCurTeamData(const ServerState & state, const nav_mesh::nav_file & navFile);
        void appendPlayerHistory();
        void clearHistory();
        int64_t getPlayerOldestContiguousHistoryIndex(int64_t playerId);
    };
}
#endif //CSKNOW_FEATURE_STORE_PRECOMMIT_H
