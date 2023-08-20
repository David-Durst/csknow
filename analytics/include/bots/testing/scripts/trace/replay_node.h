//
// Created by durst on 8/20/23.
//

#ifndef CSKNOW_REPLAY_NODE_H
#define CSKNOW_REPLAY_NODE_H

#include "bots/testing/scripts/trace/traces_data.h"
#include "bots/behavior_tree/node.h"

namespace csknow::tests::trace {
    constexpr double max_time_per_replay = 38.;

    class ReplayNode : public Node {
        const TracesData & tracesData;
        int64_t roundIndex;
        int64_t tickInRound;
        CSKnowTime roundStartTime;

    public:
        ReplayNode(Blackboard & blackboard, const TracesData & tracesData, int64_t roundIndex) :
                Node(blackboard, "Replay" + tracesData.demoFile[roundIndex]), tracesData(tracesData),
                roundIndex(roundIndex), tickInRound(0) { };

        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;

        void restart(const TreeThinker & treeThinker) override {
            tickInRound = 0;
            Node::restart(treeThinker);
        }
    };

}

#endif //CSKNOW_REPLAY_NODE_H
