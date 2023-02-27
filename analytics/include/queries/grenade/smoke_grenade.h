//
// Created by durst on 2/25/23.
//

#ifndef CSKNOW_SMOKE_GRENADE_H
#define CSKNOW_SMOKE_GRENADE_H
#include "queries/base_tables.h"


namespace csknow::smoke_grenade {
    class SmokeGrenadeResult : public QueryResult {
    public:
        vector<RangeIndexEntry> rowIndicesPerRound;
        vector<int64_t> startTickId;
        vector<int64_t> endTickId;
        vector<int64_t> tickLength;
        vector<int64_t> throwTick;
        vector<int64_t> activeTick;
        vector<int64_t> expiredTick;
        vector<int64_t> destroyTick;
        vector<int64_t> throwerId;
        vector<Vec3> pos;

        SmokeGrenadeResult() {
            variableLength = true;
            startTickColumn = 0;
            perEventLengthColumn = 2;
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
            s << index << "," << startTickId[index] << "," << endTickId[index] << "," << tickLength[index] << ","
              << throwTick[index] << "," << activeTick[index] << "," << expiredTick[index] << "," << destroyTick[index]
              << "," << throwerId[index] << "," << pos[index].toCSV();

        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"start tick id", "end tick id", "length", "throw tick id", "active tick id",
                    "expired tick id", "destroyed tick id", "thrower id"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"pos x", "pos y", "pos z"};
        }

        void runQuery(const Ticks & ticks, const Grenades & grenades,
                      const GrenadeTrajectories & grenadeTrajectories);
    };
}

#endif //CSKNOW_SMOKE_GRENADE_H
