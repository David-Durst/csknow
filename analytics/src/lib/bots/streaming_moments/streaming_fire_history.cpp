//
// Created by durst on 12/25/22.
//

#include "bots/streaming_moments/streaming_fire_history.h"

namespace csknow::fire_history {
    void StreamingFireHistory::addTickData(const StreamingBotDatabase & db) {
        const ServerState & curState = db.batchData.fromNewest();
        set<CSGOId> activeClients;

        set<CSGOId> firingClients;
        for (const auto & weaponFireEvent : curState.weaponFireEvents) {
            firingClients.insert(weaponFireEvent.shooter);
        }

        map<CSGOId, set<CSGOId>> attackerToVictims;
        for (const auto & hurtEvent : curState.hurtEvents) {
            attackerToVictims[hurtEvent.attackerId].insert(hurtEvent.victimId);
        }

        for (const auto & curTickClient : curState.clients) {
            activeClients.insert(curTickClient.csgoId);
            bool clientNew = addPlayer(curTickClient.csgoId);
            // if player is fresh or dead, give them a fresh state independent of past history
            FirePlayerData firePlayerData;
            firePlayerData.playerId = curTickClient.csgoId;
            if (db.playerHistoryLength(curTickClient.csgoId) < 2 || !curTickClient.isAlive) {
                firePlayerData.holdingAttackButton = false;
                firePlayerData.ticksSinceLastFire = MAX_TICKS_SINCE_LAST_FIRE_ATTACK;
                firePlayerData.ticksSinceLastHoldingAttack = MAX_TICKS_SINCE_LAST_FIRE_ATTACK;
                firePlayerData.hitEnemy = false;
                firePlayerData.victims = {};
            }
            // otherwise, update based on cur tick/last tick
            else {
                const ServerState::Client & priorTickClient =
                    db.batchData.fromNewest(1).getClient(curTickClient.csgoId);
                const FirePlayerData & priorFirePlayerData = firePlayerHistory.at(curTickClient.csgoId).fromNewest();

                if (firingClients.find(curTickClient.csgoId) != firingClients.end()) {
                    firePlayerData.ticksSinceLastFire = 0;
                }
                else {
                    firePlayerData.ticksSinceLastFire = std::min(MAX_TICKS_SINCE_LAST_FIRE_ATTACK,
                                                                 firePlayerData.ticksSinceLastFire + 1);
                }

                // holding attack if not reloading and recoil index going up or holding constant and greater than 0.5
                // recoil index increases by 1.0 increments, so can't be 0.5 and increasing
                // (technically could check closer to 0, but why push limits?)
                // recoil index resets to 0 on weapon switch, so anything that isn't 0 and isnt reloading
                // results from firing
                // recoil index is constant in between shots, only goes down over time
                double curRecoilIndex = curTickClient.recoilIndex;
                double priorRecoilIndex = priorTickClient.recoilIndex;
                bool isReloading = curTickClient.isReloading;
                // recoil index holds constant on some frames when decaying
                // so need to make sure equality only counts for holding if haven't started decaying
                // circular holdingAttackButton works because all initialized to false and start on second tick of history
                bool recoilIndexNotDecaying = curRecoilIndex > priorRecoilIndex ||
                                              (curRecoilIndex == priorRecoilIndex &&
                                              priorFirePlayerData.holdingAttackButton);
                firePlayerData.holdingAttackButton = !isReloading && curRecoilIndex > 0.5 && recoilIndexNotDecaying;
                if (firePlayerData.holdingAttackButton) {
                    firePlayerData.ticksSinceLastHoldingAttack = 0;
                }
                else {
                    firePlayerData.ticksSinceLastHoldingAttack =
                        std::min(MAX_TICKS_SINCE_LAST_FIRE_ATTACK, priorFirePlayerData.ticksSinceLastHoldingAttack + 1);
                }

                if (attackerToVictims.find(curTickClient.csgoId) != attackerToVictims.end()){
                    firePlayerData.hitEnemy = true;
                    firePlayerData.victims = attackerToVictims[curTickClient.csgoId];
                }
                else {
                    firePlayerData.hitEnemy = false;
                    firePlayerData.victims = {};
                }
            }
        }

        // remove no longer valid clients
        vector<CSGOId> historyClients;
        for (const auto & [csgoId, _] : firePlayerHistory) {
            historyClients.push_back(csgoId);
        }

        for (const auto & clientCSGOId : historyClients) {
            if (activeClients.find(clientCSGOId) == activeClients.end()) {
                firePlayerHistory.erase(clientCSGOId);
            }
        }
    }
}
