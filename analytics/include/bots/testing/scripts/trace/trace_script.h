//
// Created by durst on 8/20/23.
//

#ifndef CSKNOW_TRACE_SCRIPT_H
#define CSKNOW_TRACE_SCRIPT_H

#include "bots/testing/scripts/trace/traces_data.h"
#include "bots/testing/scripts/trace/replay_node.h"
#include "bots/testing/scripts/test_round.h"

namespace csknow::tests::trace {
    class TraceScript : public Script {
        const TracesData & tracesData;
        int64_t roundIndex, numRounds;
        Vec3 c4Pos;

    public:
        TraceScript(const TracesData & tracesData, int64_t roundIndex, int64_t numRounds);

        void initialize(Tree & tree, ServerState & state) override;
    };

    vector<Script::Ptr> createTracesScripts(const TracesData & tracesData, bool quitAtEnd);
}

#endif //CSKNOW_TRACE_SCRIPT_H
