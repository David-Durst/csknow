//
// Created by steam on 7/4/22.
//

#include "bots/behavior_tree/priority/memory_data.h"
#include "geometryNavConversions.h"

void EnemyPositionsMemory::updatePositions(const ServerState & state, nav_mesh::nav_file & navFile, double maxMemorySeconds) {
    vector<CSGOId> srcPlayers;

    if (considerAllTeammates) {
        srcPlayers = state.getPlayersOnTeam(team);
    }
    else {
        srcPlayers = {srcPlayer};
    }

    for (const CSGOId srcPlayer : srcPlayers) {
        if (state.getClient(srcPlayer).isAlive) {
            const auto visibleEnemies = state.getVisibleEnemies(srcPlayer);
            for (const auto & visibleEnemy : visibleEnemies) {
                positions[visibleEnemy.get().csgoId].lastSeenFootPos = visibleEnemy.get().getFootPosForPlayer();
                positions[visibleEnemy.get().csgoId].lastSeenEyePos = visibleEnemy.get().getEyePosForPlayer();
                positions[visibleEnemy.get().csgoId].lastSeenFrame = visibleEnemy.get().lastFrame;
            }
        }
    }

    // forget all enemies past length or dead
    vector<CSGOId> enemiesToForget;
    int32_t curFrame = state.getLastFrame();
    for (const auto & [id, position] : positions) {
        double timeSinceSeen = state.getSecondsBetweenFrames(curFrame, position.lastSeenFrame);
        if (timeSinceSeen > maxMemorySeconds || !state.getClient(id).isAlive) {
            enemiesToForget.push_back(id);
        }
    }
    for (const auto & enemyToForget : enemiesToForget) {
        positions.erase(enemyToForget);
    }
}
