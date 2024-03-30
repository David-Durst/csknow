//
// Created by durst on 3/29/24.
//
#include <regex>
#include "bots/testing/scripts/test_survey.h"

namespace csknow::survey {
    const vector<string> actualBotNames{"Learned", "Learned Combat", "Hand-Crafted", "Default"};

    void SurveyScript::initialize(Tree &tree, ServerState &state) {
        RoundScript::initialize(tree, state);
        if (tree.newBlackboard) {
            Blackboard &blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Node::Ptr oldCommands{std::move(commands)};

            // start commands
            string scenarioInstructions = "STARTING SCENARIO " + std::to_string(scenarioIndex);
            string botInstructions = "Playing scenario " + std::to_string(scenarioIndex) + ", bot " +
                    std::to_string(botIndex) + " in 3 seconds";
            vector<Node::Ptr> setupNodes = Node::makeList(
                    make_unique<SayCmd>(blackboard, botInstructions),
                    make_unique<movement::WaitNode>(blackboard, 3)
            );
            if (botIndex == 0) {
                setupNodes.insert(setupNodes.begin(), make_unique<SayCmd>(blackboard, scenarioInstructions));
            }
            if (botScenarioOrder[botIndex] == BotType::Default) {
                setupNodes.insert(setupNodes.begin(), make_unique<SetBotStop>(blackboard, "0"));
            }
            else {
                setupNodes.insert(setupNodes.begin(), make_unique<SetBotStop>(blackboard, "1"));
                if (botScenarioOrder[botIndex] == BotType::Handcrafted) {
                    setupNodes.insert(setupNodes.begin(), make_unique<SetUseLearnedModel>(blackboard, false, ENGINE_TEAM_T));
                    setupNodes.insert(setupNodes.begin(), make_unique<SetUseLearnedModel>(blackboard, false, ENGINE_TEAM_CT));
                }
                else {
                    setupNodes.insert(setupNodes.begin(), make_unique<SetUseLearnedModel>(blackboard, true, ENGINE_TEAM_T));
                    setupNodes.insert(setupNodes.begin(), make_unique<SetUseLearnedModel>(blackboard, true, ENGINE_TEAM_CT));
                }
            }
            Node::Ptr setup = make_unique<SequenceNode>(blackboard, std::move(setupNodes));
            vector<Node::Ptr> newCommandNodes = Node::makeList(
                    std::move(setup),
                    std::move(oldCommands)
            );

            // end commands
            if (botIndex == static_cast<int>(actualBotNames.size())) {
                string rankingInstructions = "Please rank the bots in this scenario based on the following statement: "
                                             "Player movement matches your expectation of how humans would move in the scenario situation. "
                                             "Please rank bots from best match of your expectations to worst match using a comma seperated list like 1,2,3,4";
                Node::Ptr finish = make_unique<SequenceNode>(blackboard, Node::makeList(
                        make_unique<SayCmd>(blackboard, rankingInstructions),
                        make_unique<CollectBotRankingCommand>(blackboard, scenarioIndex, botScenarioOrder)
                ));
                newCommandNodes.push_back(std::move(finish));
            }
            commands = make_unique<SequenceNode>(blackboard, std::move(newCommandNodes));
        }
    }

    std::regex ranking_regex("[1-4],[1-4],[1-4],[1-4]",
                             std::regex_constants::ECMAScript | std::regex_constants::icase);
    string botTypeToString(BotType botType) {
        if (botType == BotType::Learned) {
            return "Learned";
        }
        else if (botType == BotType::LearnedCombat) {
            return "LearnedCombat";
        }
        else if (botType == BotType::Handcrafted) {
            return "Hand-Crafted";
        }
        else {
            return "Default";
        }
    }

    NodeState CollectBotRankingCommand::exec(const ServerState &state, TreeThinker &treeThinker) {
        bool foundRanking = false;
        for (const auto & sayEvent : state.sayEvents) {
            std::smatch ranking_match;
            if (std::regex_search(sayEvent.message, ranking_match, ranking_regex)) {
                vector<string> botRanking;
                vector<bool> foundAllBots{false, false, false, false};
                for (const auto & m : ranking_match) {
                    int botIndex = std::stoi(m.str()) - 1;
                    foundAllBots[botIndex] = true;
                    botRanking.push_back(botTypeToString(botScenarioOrder[botIndex]));
                }

                if (foundAllBots[0] && foundAllBots[1] && foundAllBots[2] && foundAllBots[3]) {
                    foundRanking = true;

                    // create file if not exists
                    string playerName = state.getClient(sayEvent.player).name;
                    string playerFile = "/home/durst/responses/" + playerName + ".csv";
                    if (!std::filesystem::exists(playerFile)) {
                        std::ofstream headerStream(playerFile);
                        headerStream << "name,scenario id,ranking" << std::endl;
                        headerStream.close();
                    }

                    // append result to file for player
                    std::ofstream resultStream(playerFile, std::ofstream::app);
                    resultStream << playerName << "," << scenarioId << ",";
                    for (size_t i = 0; i < botRanking.size(); i++) {
                        if (i > 0) {
                            resultStream << ";";
                        }
                        resultStream << botRanking[i];
                    }
                    resultStream << std::endl;
                    resultStream.close();

                    break;
                }
            }
        }

        if (foundRanking) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }

        return playerNodeState[treeThinker.csgoId];
    }

    NodeState SetUseLearnedModel::exec(const ServerState &, TreeThinker &treeThinker) {
        setAllTeamModelProbabilities(useLearnedModel, teamId);
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

    vector<Script::Ptr> createSurveyScripts(const csknow::plant_states::PlantStatesResult & plantStatesResult,
                                            int startSituationId, bool quitAtEnd, int numHumans) {
        vector<Script::Ptr> result;

        std::random_device rd;
        std::mt19937 gen;
        std::uniform_real_distribution<> dis;

        size_t numRounds = static_cast<size_t>(plantStatesResult.size);

        size_t batchSize = 100;
        size_t maxI = std::min(batchSize * (startSituationId + 1), numRounds);
        if (startSituationId == -1) {
            startSituationId = 0;
            maxI = numRounds;
        }
        //maxI = 100;
        for (size_t i = batchSize * startSituationId; i < maxI; i++) {
            vector<BotType> botScenarioOrder{BotType::Learned, BotType::LearnedCombat, BotType::Handcrafted, BotType::Default};
            shuffle(botScenarioOrder.begin(), botScenarioOrder.end(), gen);
            for (size_t j = 0; j < enumAsInt(BotType::NUM_BOT_TYPES); j++) {
                result.push_back(make_unique<SurveyScript>(plantStatesResult, i, maxI, gen, dis,
                                                           std::nullopt, "RoundScript", std::nullopt, std::nullopt,
                                                           numHumans, j, botScenarioOrder));
            }
        }
        if (quitAtEnd) {
            result.push_back(make_unique<QuitScript>());
        }
        else {
            result.push_back(make_unique<WaitUntilScoreScript>());
        }

        return result;

    }

}
