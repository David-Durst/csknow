//
// Created by durst on 8/20/23.
//

#ifndef CSKNOW_TRACE_SCRIPT_H
#define CSKNOW_TRACE_SCRIPT_H

#include "bots/testing/scripts/trace/traces_data.h"
#include "bots/testing/scripts/trace/replay_node.h"
#include "bots/testing/scripts/test_round.h"

namespace csknow::tests::trace {
    constexpr int num_trace_repeats = 10;

    class TraceScript : public Script {
        const TracesData & tracesData;
        int64_t traceIndex, numTraces;
        int64_t roundIndex, numRounds;
        Vec3 c4Pos;
        bool oneTeam;
        bool oneBot;
        string botStop;
        vector<Vec3> startPos;
        vector<Vec2> startViewAngle;

    public:
        TraceScript(const TracesData & tracesData, int64_t traceIndex, int64_t numTraces, int64_t roundIndex,
                    int64_t numRounds, bool oneBot, bool oneTeam, const string & botStop);

        void computeInitialPositionsViewAngles();
        void initialize(Tree & tree, ServerState & state) override;
    };

    vector<Script::Ptr> createTracesScripts(const TracesData & tracesData, const string & botStop, bool quitAtEnd);
}

#endif //CSKNOW_TRACE_SCRIPT_H
