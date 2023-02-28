//
// Created by durst on 2/27/23.
//

#ifndef CSKNOW_PLAYER_FLASHED_H
#define CSKNOW_PLAYER_FLASHED_H
#include "queries/base_tables.h"
#include "enum_helpers.h"

namespace csknow::player_flashed {
    class PlayerFlashedResult : public QueryResult {
    public:
        vector<RangeIndexEntry> rowIndicesPerRound;
        vector<int64_t> tickId;
        vector<int64_t> victimId;
        vector<double> flashAmount;
        IntervalIndex playerFlashedPerTick;

        PlayerFlashedResult() {
            variableLength = false;
            startTickColumn = 0;
            ticksPerEvent = 1;
        }

        vector<int64_t> filterByForeignKey(int64_t otherTableIndex) override {
            vector<int64_t> result;
            for (int64_t i = rowIndicesPerRound[otherTableIndex].minId; i <= rowIndicesPerRound[otherTableIndex].maxId; i++) {
                if (i == -1) {
                    continue;
                }
                result.push_back(i);
            }
            return result;
        }

        void oneLineToCSV(int64_t index, std::ostream &s) override {
            s << index << "," << tickId[index] << "," << victimId[index]
              << "," << flashAmount[index]
              << std::endl;

        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"tick id", "victim id"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"flash amount"};
        }

        void runQuery(const Games & games, const Rounds & rounds, const Ticks & ticks,
                      const PlayerAtTick & playerAtTick, const Flashed & flashed);
    };
}

#endif //CSKNOW_PLAYER_FLASHED_H
