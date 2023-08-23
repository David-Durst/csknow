//
// Created by durst on 8/20/23.
//

#include "bots/testing/scripts/trace/trace_script.h"
#include "bots/testing/scripts/learned/log_nodes.h"

namespace csknow::tests::trace {
    TraceScript::TraceScript(const csknow::tests::trace::TracesData &tracesData, int64_t roundIndex,
                             int64_t numRounds, bool oneTeam, bool oneBot) :
                             Script("TraceScript", {}, {ObserveType::FirstPerson, 0}), tracesData(tracesData),
                             roundIndex(roundIndex), numRounds(numRounds), oneTeam(oneTeam), oneBot(oneBot) {
        int64_t tickInFeatureStore = tracesData.startIndices[roundIndex];

        name += tracesData.demoFile[roundIndex] + "_" + std::to_string(tracesData.roundNumber[roundIndex]) +
                "_" + std::to_string(oneTeam) + "_" + std::to_string(oneBot);
        int numCT = 0, numT = 0;
        neededBots.clear();

        c4Pos = tracesData.teamFeatureStoreResult.c4Pos[tickInFeatureStore];
        map<int64_t, int64_t> ctFeatureStoreIndexToBotIndex, tFeatureStoreIndexToBotIndex;
        int64_t anAliveCTFeatureStoreIndex, anAliveTFeatureStoreIndex;
        for (size_t i = 0; i < feature_store::max_enemies; i++) {
            if (tracesData.teamFeatureStoreResult.columnCTData[i].alive[tickInFeatureStore] && numCT < maxCT) {
                ctFeatureStoreIndexToBotIndex[static_cast<int64_t>(i)] = static_cast<int64_t>(neededBots.size());
                anAliveCTFeatureStoreIndex = static_cast<int64_t>(i);
                numCT++;
                neededBots.push_back({0, ENGINE_TEAM_CT});
            }
            if (tracesData.teamFeatureStoreResult.columnTData[i].alive[tickInFeatureStore] && numT < maxT) {
                tFeatureStoreIndexToBotIndex[static_cast<int64_t>(i)] = static_cast<int64_t>(neededBots.size());
                anAliveTFeatureStoreIndex = static_cast<int64_t>(i);
                numT++;
                neededBots.push_back({0, ENGINE_TEAM_T});
            }
        }

        // if one team is bots, spectate a bot from that team
        // if one player is a bot, spectate that bot
        if (oneTeam) {
            if (oneBot) {
                if (tracesData.ctBot[roundIndex]) {
                    observeSettings.neededBotIndex =
                            ctFeatureStoreIndexToBotIndex[tracesData.oneBotFeatureStoreIndex[roundIndex]];
                }
                else {
                    observeSettings.neededBotIndex =
                            tFeatureStoreIndexToBotIndex[tracesData.oneBotFeatureStoreIndex[roundIndex]];
                }
            }
            else {
                if (tracesData.ctBot[roundIndex]) {
                    observeSettings.neededBotIndex = ctFeatureStoreIndexToBotIndex[anAliveCTFeatureStoreIndex];
                }
                else {
                    observeSettings.neededBotIndex = tFeatureStoreIndexToBotIndex[anAliveTFeatureStoreIndex];
                }
            }
        }
    }

    void TraceScript::initialize(Tree &tree, ServerState &state) {
        if (tree.newBlackboard) {
            Blackboard &blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            vector<CSGOId> neededBotIds = getNeededBotIds();
            bool lastRound = numRounds == roundIndex + 1;

            vector<string> nonReplayNames;
            for (size_t i = 0; i < neededBotIds.size(); i++) {
                const auto & client = state.getClient(neededBotIds[i]);
                if (oneTeam) {
                    if (tracesData.ctBot[roundIndex] && client.team == ENGINE_TEAM_CT) {
                        if (!oneBot || observeSettings.neededBotIndex == static_cast<int64_t>(i)) {
                            nonReplayNames.push_back(client.name);
                        }
                    }
                    else if (!tracesData.ctBot[roundIndex] && client.team == ENGINE_TEAM_T) {
                        if (!oneBot || observeSettings.neededBotIndex == static_cast<int64_t>(i)) {
                            nonReplayNames.push_back(client.name);
                        }
                    }
                }
            }

            stringstream strStream;
            commaSeparateList(strStream, nonReplayNames);

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
                    make_unique<ReplayNode>(blackboard, tracesData, roundIndex, oneTeam, oneBot)));

            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<csknow::tests::learned::StartNode>(blackboard, name, roundIndex, numRounds),
                                                         make_unique<SayCmd>(blackboard, "non_replay_players," + strStream.str()),
                                                         std::move(bodyNode),
                                                         make_unique<csknow::tests::learned::SuccessEndNode>(blackboard, name, roundIndex, numRounds)),
                                                 "RoundSequence");
        }
    }

    vector<Script::Ptr> createTracesScripts(const TracesData & tracesData, bool quitAtEnd) {
        vector<Script::Ptr> result;

        int64_t numRounds = static_cast<int64_t>(tracesData.demoFile.size());
        for (int64_t i = 0; i < numRounds; i++) {
            for (int j = 0; j < num_trace_repeats; j++) {
                result.push_back(make_unique<TraceScript>(tracesData, i, numRounds, true, false));
                result.push_back(make_unique<TraceScript>(tracesData, i, numRounds, true, true));
            }
        }
        if (quitAtEnd) {
            result.push_back(make_unique<QuitScript>());
        } else {
            result.push_back(make_unique<WaitUntilScoreScript>());
        }

        return result;
    }
}
