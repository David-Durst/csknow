//
// Created by durst on 2/25/23.
//

#ifndef CSKNOW_SMOKE_GRENADE_H
#define CSKNOW_SMOKE_GRENADE_H
#include "queries/base_tables.h"
#include "enum_helpers.h"


namespace csknow::smoke_grenade {
    enum class SmokeGrenadeState {
        Thrown,
        Active
    };

    class SmokeGrenadeResult : public QueryResult {
    public:
        vector<RangeIndexEntry> rowIndicesPerRound;
        vector<int64_t> tickId;
        vector<int64_t> throwerId;
        vector<SmokeGrenadeState> state;
        vector<Vec3> pos;
        IntervalIndex smokeGrenadesPerTick;

        SmokeGrenadeResult() {
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
            s << index << "," << tickId[index] << "," << throwerId[index]
              << "," << enumAsInt(state[index]) << "," << pos[index].toCSV()
              << std::endl;

        }

        [[nodiscard]]
        vector<string> getForeignKeyNames() override {
            return {"tick id", "thrower id"};
        }

        [[nodiscard]]
        vector<string> getOtherColumnNames() override {
            return {"state", "pos x", "pos y", "pos z"};
        }

        void runQuery(const Rounds & rounds, const Ticks & ticks, const Grenades & grenades,
                      const GrenadeTrajectories & grenadeTrajectories);
    };
}

#endif //CSKNOW_SMOKE_GRENADE_H
