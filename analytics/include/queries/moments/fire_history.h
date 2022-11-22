//
// Created by durst on 7/6/22.
//

#ifndef CSKNOW_FIRE_HISTORY_H
#define CSKNOW_FIRE_HISTORY_H
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "load_data.h"
#include "queries/query.h"
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::map;

namespace csknow::fire_history {
    class FireHistoryResult : public QueryResult {
    public:
        const Rounds & rounds;
        const Ticks & ticks;
        vector<RangeIndexEntry> rowIndicesPerRound;
        vector<int64_t> tickId;
        vector<int64_t> playerId;
        vector<int64_t> holdingAttackButton;
        vector<int64_t> ticksSinceLastFire;
        vector<int64_t> ticksSinceLastHoldingAttack;
        vector<int64_t> ticksUntilNextFire;
        vector<int64_t> ticksUntilNextHoldingAttack;
        vector<bool> hitEnemy;
        vector<set<int64_t>> victims;

        FireHistoryResult(const Rounds & rounds, const Ticks & ticks) :
                rounds(rounds), ticks(ticks) {
            variableLength = false;
            startTickColumn = -1;
        }

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
            vector<int64_t> result;
            for (int64_t i = rounds.ticksPerRound[otherTableIndex].minId;
                 i <= rounds.ticksPerRound[otherTableIndex].maxId; i++) {
                for (int64_t j = ticks.patPerTick[i].minId; j <= ticks.patPerTick[i].maxId; j++) {
                    if (j != -1) {
                        result.push_back(j);
                    }
                }
            }
            return result;
        }

        void oneLineToCSV(int64_t index, stringstream & ss) override {
            ss << index << "," << tickId[index] << "," << playerId[index]
               << "," << holdingAttackButton[index]
               << "," << ticksSinceLastFire[index] << "," << ticksSinceLastHoldingAttack[index]
               << "," << ticksUntilNextFire[index] << "," << ticksUntilNextHoldingAttack[index];
            ss << std::endl;
        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"tick id", "player id"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"ticks since last fire", "ticks since last holding attack", "ticks until next fire",
                    "ticks until next holding attack"};
        }

        void runQuery(const Games & games, const WeaponFire & weaponFire, const Hurt & hurt,
                      const PlayerAtTick & playerAtTick);
    };
}

#endif //CSKNOW_FIRE_HISTORY_H
