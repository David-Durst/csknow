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
        vector<int64_t> tickId;
        vector<int64_t> playerId;
        vector<int16_t> ticksSinceLastFire;
        vector<int64_t> lastShotFiredTickId;
        vector<int16_t> ticksUntilNextFire;
        vector<int64_t> nextShotFiredTickId;
        vector<int64_t> holdingAttackButton;

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
               << "," << ticksSinceLastFire[index] << "," << lastShotFiredTickId[index]
               << "," << holdingAttackButton[index];
            ss << std::endl;
        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"tick id", "player id"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"ticks since last fire", "last shot fired tick id", "holding attack button"};
        }

        void runQuery(const WeaponFire & weaponFire, const PlayerAtTick & playerAtTick);
    };
}

#endif //CSKNOW_FIRE_HISTORY_H
