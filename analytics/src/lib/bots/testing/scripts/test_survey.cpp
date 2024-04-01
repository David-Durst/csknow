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
                    std::to_string(botIndex + 1) + " in 3 seconds";
            vector<Node::Ptr> setupNodes;

            // optional experience collection
            // assuming only one player is playing at a time
            string playerName;
            for (const auto & client : state.clients) {
                if (!client.isBot && (client.team == ENGINE_TEAM_T || client.team == ENGINE_TEAM_CT)) {
                    playerName = client.name;
                    break;
                }
            }
            // only ask for csgo experience if not provided yet
            string csgoExperienceInstructions =
                    "What is the highest rank you achieved in CSGO matchmaking? Please answer with one of the "
                    "following abbreviations. Please do not include CS2, Faceit, or ESEA ranks. Please answer with NA "
                    "if you never had a CSGO matchmaking rank.: ";
            string csgoExperienceOptions =
                    "NA, S1, S2, S3, S4, SE, SEM, GN1, GN2, GN3, GNM, MG1, MG2, MGE, DMG, LE, LEM, SMFC, GE";
            setupNodes.push_back(make_unique<SelectorNode>(blackboard, Node::makeList(
                    make_unique<CheckForSentinelFile>(blackboard, "/home/steam/responses/" + playerName + "_csgo_experience.txt"),
                    make_unique<SequenceNode>(blackboard, Node::makeList(
                        make_unique<SayCmd>(blackboard, csgoExperienceInstructions),
                        make_unique<SayCmd>(blackboard, csgoExperienceOptions),
                        make_unique<CollectCSGOExperienceCommand>(blackboard, scenarioIndex)
                    ))
            )));
            // same for dev experience
            string devExperienceInstructions =
                    "How many years of professional experience do you have working on AAA FPS games with a multiplayer "
                    "component? Please answer with a number like 0 or 5 or 15.";
            setupNodes.push_back(make_unique<SelectorNode>(blackboard, Node::makeList(
                    make_unique<CheckForSentinelFile>(blackboard, "/home/steam/responses/" + playerName + "_dev_experience.txt"),
                    make_unique<SequenceNode>(blackboard, Node::makeList(
                            make_unique<SayCmd>(blackboard, devExperienceInstructions),
                            make_unique<CollectDevExperienceCommand>(blackboard, scenarioIndex)
                    ))
            )));

            // scenario instructions
            if (botIndex == 0) {
                setupNodes.push_back(make_unique<SayCmd>(blackboard, scenarioInstructions));
            }
            bool printBotNames = false;
            if (botScenarioOrder[botIndex] == BotType::Default) {
                setupNodes.push_back(make_unique<SetBotStop>(blackboard, "0"));
                if (printBotNames) {
                    setupNodes.push_back(make_unique<SayCmd>(blackboard, "default"));
                }
            }
            else {
                setupNodes.push_back(make_unique<SetBotStop>(blackboard, "1"));
                if (botScenarioOrder[botIndex] == BotType::Handcrafted) {
                    setupNodes.push_back(make_unique<SetUseLearnedModel>(blackboard, false, ENGINE_TEAM_T));
                    setupNodes.push_back(make_unique<SetUseLearnedModel>(blackboard, false, ENGINE_TEAM_CT));
                    if (printBotNames) {
                        setupNodes.push_back(make_unique<SayCmd>(blackboard, "hand-crafted"));
                    }
                }
                else {
                    setupNodes.push_back(make_unique<SetUseLearnedModel>(blackboard, true, ENGINE_TEAM_T));
                    setupNodes.push_back(make_unique<SetUseLearnedModel>(blackboard, true, ENGINE_TEAM_CT));
                    if (botScenarioOrder[botIndex] == BotType::Learned) {
                        setupNodes.push_back(make_unique<SetUseUncertainModel>(blackboard, false));
                        if (printBotNames) {
                            setupNodes.push_back(make_unique<SayCmd>(blackboard, "learned"));
                        }
                    }
                    else if (botScenarioOrder[botIndex] == BotType::LearnedCombat) {
                        setupNodes.push_back(make_unique<SetUseUncertainModel>(blackboard, true));
                        if (printBotNames) {
                            setupNodes.push_back(make_unique<SayCmd>(blackboard, "learned combat"));
                        }
                    }
                }
            }
            // bot-specific instructions
            setupNodes.push_back(make_unique<SayCmd>(blackboard, botInstructions));
            setupNodes.push_back(make_unique<movement::WaitNode>(blackboard, 3));

            Node::Ptr setup = make_unique<SequenceNode>(blackboard, std::move(setupNodes));
            vector<Node::Ptr> newCommandNodes = Node::makeList(
                    std::move(setup),
                    std::move(oldCommands)
            );

            // end commands
            if (botIndex == static_cast<int>(actualBotNames.size() - 1)) {
                string rankingInstructions0 = "Please rank the bots in this scenario based on the following statement: "
                                             "Player movement matches your expectation of how humans would move in the scenario situation. ";
                string rankingInstructions1 = "Please rank bots from best match of your expectations to worst match using a comma seperated list like 1,2,3,4";
                Node::Ptr finish = make_unique<SequenceNode>(blackboard, Node::makeList(
                        make_unique<SayCmd>(blackboard, rankingInstructions0),
                        make_unique<SayCmd>(blackboard, rankingInstructions1),
                        make_unique<CollectBotRankingCommand>(blackboard, scenarioIndex, botScenarioOrder),
                        make_unique<SetBotStop>(blackboard, "1"),
                        make_unique<SetUseLearnedModel>(blackboard, true, ENGINE_TEAM_T),
                        make_unique<SetUseLearnedModel>(blackboard, true, ENGINE_TEAM_CT),
                        make_unique<SetUseUncertainModel>(blackboard, true)
                ));
                newCommandNodes.push_back(std::move(finish));
            }
            commands = make_unique<SequenceNode>(blackboard, std::move(newCommandNodes));
        }
    }

    string getUserFilePath(const ServerState &state, CSGOId playerId) {
        string playerName = state.getClient(playerId).name;
        string playerFile = "/home/steam/responses/" + playerName + ".csv";
        if (!std::filesystem::exists(playerFile)) {
            std::ofstream headerStream(playerFile);
            headerStream << "name,time,scenario id,response type,payload" << std::endl;
            headerStream.close();
        }
        return playerFile;
    }

    string getNowAsISOString() {
        std::time_t t
                = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::stringstream ss;
        ss << std::put_time( std::localtime( &t ), "%FT%T%z");
        return ss.str();
    }

    NodeState CheckForSentinelFile::exec(const ServerState &, TreeThinker &treeThinker) {
        if (std::filesystem::exists(filePath)) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }

        return playerNodeState[treeThinker.csgoId];
    }

    // need to put longer strings before shorter ones so substring match doesn't find shorter one in longer one
    vector<string> csgoExperience = {"NA", "S1", "S2", "S3", "S4", "SEM", "SE", "GN1", "GN2", "GN3", "GNM",
                                     "MG1", "MG2", "MGE", "DMG", "LEM", "LE", "SMFC", "GE"};

    NodeState CollectCSGOExperienceCommand::exec(const ServerState &state, TreeThinker &treeThinker) {
        bool foundCSGOExperience = false;
        for (const auto & sayEvent : state.sayEvents) {
            if (sayEvent.player == 0) {
                continue;
            }
            std::smatch ranking_match;
            for (const auto & c : csgoExperience) {
                if (sayEvent.message.find(c) != std::string::npos) {
                    foundCSGOExperience = true;

                    string playerName = state.getClient(sayEvent.player).name;

                    // create indicator that player experience recorded
                    string sentinelFile = "/home/steam/responses/" + playerName + "_csgo_experience.txt";
                    std::ofstream sentinelStream(sentinelFile);
                    sentinelStream << std::endl;
                    sentinelStream.close();

                    // create file if not exists
                    string playerFile = getUserFilePath(state, sayEvent.player);

                    // append result to file for player
                    std::ofstream resultStream(playerFile, std::ofstream::app);
                    resultStream << playerName << "," << getNowAsISOString() << ","
                                 << scenarioId << "," << "CSGO experience" << "," << c << std::endl;
                    resultStream.close();
                    break;
                }
            }
        }

        if (foundCSGOExperience) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }

        return playerNodeState[treeThinker.csgoId];
    }

    std::regex devExperienceRegex("[0-9]+",
                             std::regex_constants::ECMAScript | std::regex_constants::icase);

    NodeState CollectDevExperienceCommand::exec(const ServerState &state, TreeThinker &treeThinker) {
        bool foundDevExperience = false;
        for (const auto & sayEvent : state.sayEvents) {
            if (sayEvent.player == 0) {
                continue;
            }
            std::smatch devExperienceMatch;
            if (std::regex_search(sayEvent.message, devExperienceMatch, devExperienceRegex)) {
                foundDevExperience = true;
                int devExperience = std::stoi(devExperienceMatch.str());

                string playerName = state.getClient(sayEvent.player).name;

                // create indicator that player experience recorded
                string sentinelFile = "/home/steam/responses/" + playerName + "_dev_experience.txt";
                std::ofstream sentinelStream(sentinelFile);
                sentinelStream << std::endl;
                sentinelStream.close();

                // create results file if not exists
                string playerFile = getUserFilePath(state, sayEvent.player);

                // append result to file for player
                std::ofstream resultStream(playerFile, std::ofstream::app);
                resultStream << playerName << "," << getNowAsISOString() << ","
                             << scenarioId << "," << "dev experience" << "," << devExperience;
                resultStream << std::endl;
                resultStream.close();

                break;
            }
        }

        if (foundDevExperience) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }

        return playerNodeState[treeThinker.csgoId];
    }


    std::regex rankingRegex("([1-4]),([1-4]),([1-4]),([1-4])",
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
            if (sayEvent.player == 0) {
                continue;
            }
            std::smatch rankingMatch;
            if (std::regex_search(sayEvent.message, rankingMatch, rankingRegex)) {
                vector<string> botRanking;
                vector<bool> foundAllBots{false, false, false, false};
                for (size_t i = 1; i < rankingMatch.size(); i++) {
                    int botIndex = std::stoi(rankingMatch[i].str()) - 1;
                    foundAllBots[botIndex] = true;
                    botRanking.push_back(botTypeToString(botScenarioOrder[botIndex]));
                }

                if (foundAllBots[0] && foundAllBots[1] && foundAllBots[2] && foundAllBots[3]) {
                    foundRanking = true;

                    // create file if not exists
                    string playerName = state.getClient(sayEvent.player).name;
                    string playerFile = getUserFilePath(state, sayEvent.player);

                    // append result to file for player
                    std::ofstream resultStream(playerFile, std::ofstream::app);
                    resultStream << playerName << "," << getNowAsISOString() << ","
                        << scenarioId << "," << "ranking" << ",";
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

    NodeState SetUseUncertainModel::exec(const ServerState &, TreeThinker &treeThinker) {
        setUseUncertainModel(useUncertainModel);
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

    vector<Script::Ptr> createSurveyScripts(const csknow::plant_states::PlantStatesResult & plantStatesResult,
                                            int startSituationId, bool quitAtEnd, int numHumans) {
        vector<Script::Ptr> result;

        std::random_device rd;
        std::mt19937 gen(rd());
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
            for (size_t j = 0; j < static_cast<size_t>(BotType::NUM_BOT_TYPES); j++) {
                result.push_back(make_unique<SurveyScript>(plantStatesResult, i, i * enumAsInt(BotType::NUM_BOT_TYPES) + j,
                                                           maxI * enumAsInt(BotType::NUM_BOT_TYPES), gen, dis,
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
