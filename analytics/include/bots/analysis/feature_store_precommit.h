//
// Created by durst on 4/6/23.
//

#ifndef CSKNOW_FEATURE_STORE_PRECOMMIT_H
#define CSKNOW_FEATURE_STORE_PRECOMMIT_H

#include "queries/query.h"
#include "bots/load_save_bot_data.h"
#include "geometryNavConversions.h"
#include <map>

namespace csknow::feature_store {
    struct BTTeamPlayerData {
        int64_t playerId;
        TeamId teamId;
        AreaId curArea;
        int64_t curAreaIndex;
    };

    struct C4MapData {
        bool c4Planted;
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
    };
}
#endif //CSKNOW_FEATURE_STORE_PRECOMMIT_H
