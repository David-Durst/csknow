//
// Created by durst on 8/20/23.
//

#ifndef CSKNOW_REPLAY_NODE_H
#define CSKNOW_REPLAY_NODE_H

#include "bots/testing/scripts/trace/traces_data.h"
#include "bots/behavior_tree/node.h"
#include "bots/testing/script_data.h"

namespace csknow::tests::trace {
    constexpr double max_time_per_replay = 38.;

    class ReplayNode : public Node {
        const TracesData & tracesData;
        vector<NeededBot> neededBots;
        int64_t traceIndex;
        int64_t curRoundTick = 0;
        bool oneTeam;
        bool oneBot;
        CSKnowTime roundStartTime;
        int64_t startFrame = 0;

    public:
        ReplayNode(Blackboard & blackboard, const TracesData & tracesData, const vector<NeededBot> & neededBots,
                   int64_t traceIndex, bool oneTeam, bool oneBot) :
                Node(blackboard, "Replay" + tracesData.demoFile[traceIndex]), tracesData(tracesData),
                neededBots(neededBots), traceIndex(traceIndex), oneTeam(oneTeam), oneBot(oneBot) { };

        NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;

        void restart(const TreeThinker & treeThinker) override {
            curRoundTick = 0;
            Node::restart(treeThinker);
        }
    };

}

#endif //CSKNOW_REPLAY_NODE_H
