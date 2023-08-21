//
// Created by durst on 8/20/23.
//

#include "bots/testing/scripts/trace/trace_script.h"
#include "bots/testing/scripts/learned/log_nodes.h"

namespace csknow::tests::trace {
    TraceScript::TraceScript(const csknow::tests::trace::TracesData &tracesData, int64_t roundIndex,
                             int64_t numRounds) :
                             Script("TraceScript", {}, {ObserveType::FirstPerson, 0}), tracesData(tracesData),
                             roundIndex(roundIndex), numRounds(numRounds) {
        int64_t tickInFeatureStore = tracesData.startIndices[roundIndex];

        name += tracesData.demoFile[roundIndex];
        int numCT = 0, numT = 0;
        neededBots.clear();

        c4Pos = tracesData.teamFeatureStoreResult.c4Pos[tickInFeatureStore];
        for (size_t i = 0; i < feature_store::max_enemies; i++) {
            if (tracesData.teamFeatureStoreResult.columnCTData[i].alive[tickInFeatureStore] && numCT < maxCT) {
                numCT++;
                neededBots.push_back({0, ENGINE_TEAM_CT});
            }
            if (tracesData.teamFeatureStoreResult.columnTData[i].alive[tickInFeatureStore] && numT < maxT) {
                numT++;
                neededBots.push_back({0, ENGINE_TEAM_T});
            }
        }
    }

    void TraceScript::initialize(Tree &tree, ServerState &state) {
        if (tree.newBlackboard) {
            Blackboard &blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            vector<CSGOId> neededBotIds = getNeededBotIds();
            bool lastRound = numRounds == roundIndex + 1;
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<InitGameRound>(blackboard, name),
                    make_unique<SetMaxRounds>(blackboard, lastRound ? 2 : 20, true),
                    make_unique<movement::WaitNode>(blackboard, 0.3),
                    make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                    make_unique<SlayAllBut>(blackboard, neededBotIds, state),
                    make_unique<SetPos>(blackboard, c4Pos, Vec2({0., 0.})),
                    make_unique<TeleportPlantedC4>(blackboard),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                    make_unique<RecomputeOrdersNode>(blackboard)), "RoundSetup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", neededBotIds, false)
            ), "DefuserDisableDuringSetup");

            Node::Ptr bodyNode = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    make_unique<RepeatDecorator>(blackboard, make_unique<RoundStart>(blackboard), true),
                    make_unique<ReplayNode>(blackboard, tracesData, roundIndex)));

            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<csknow::tests::learned::StartNode>(blackboard, name, roundIndex, numRounds),
                                                         std::move(bodyNode),
                                                         make_unique<csknow::tests::learned::SuccessEndNode>(blackboard, name, roundIndex, numRounds)),
                                                 "RoundSequence");
        }
    }

    vector<Script::Ptr> createTracesScripts(const TracesData & tracesData, bool quitAtEnd) {
        vector<Script::Ptr> result;

        int64_t numRounds = 300;/*static_cast<int64_t>(tracesData.demoFile.size());*/
        for (int64_t i = 0; i < numRounds; i++) {
            result.push_back(make_unique<TraceScript>(tracesData, 0, numRounds));
        }
        if (quitAtEnd) {
            result.push_back(make_unique<QuitScript>());
        } else {
            result.push_back(make_unique<WaitUntilScoreScript>());
        }

        return result;
    }
}
