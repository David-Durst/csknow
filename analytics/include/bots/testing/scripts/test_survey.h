//
// Created by durst on 3/29/24.
//

#ifndef CSKNOW_TEST_SURVEY_H
#define CSKNOW_TEST_SURVEY_H

#include "bots/analysis/learned_models.h"
#include "bots/testing/scripts/test_round.h"
#include <array>

namespace csknow::survey {
    enum class BotType {
        Learned = 0,
        LearnedCombat = 1,
        Handcrafted = 2,
        Default = 3,
        NUM_BOT_TYPES
    };

    class SurveyScript : public RoundScript {
        int scenarioIndex, botIndex;
        vector<BotType> botScenarioOrder;
    public:
        SurveyScript(const csknow::plant_states::PlantStatesResult & plantStatesResult, size_t plantStateIndex,
                     size_t roundIndex, size_t numRounds, std::mt19937 gen, std::uniform_real_distribution<> dis,
                     std::optional<vector<bool>> playerFreeze, string baseName, std::optional<Vec3> cameraOrigin,
                     std::optional<Vec2> cameraAngle, int numHumans, int botIndex,
                     const vector<BotType> & botScenarioOrder) :
                     RoundScript(plantStatesResult, plantStateIndex, roundIndex, numRounds, gen, dis,
                                 playerFreeze, baseName, cameraOrigin, cameraAngle, numHumans),
                                 scenarioIndex(plantStateIndex), botIndex(botIndex),
                                 botScenarioOrder(botScenarioOrder) { };

        void initialize(Tree & tree, ServerState & state) override;
    };

    struct CheckForSentinelFile : Node {
        string filePath;
        CheckForSentinelFile(Blackboard & blackboard, string filePath) :
            Node(blackboard, "CheckForSentinelFile"), filePath(filePath) { };

        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    struct CollectCSGOExperienceCommand : Node {
        int scenarioId;
        CollectCSGOExperienceCommand(Blackboard & blackboard, int scenarioId) :
            Node(blackboard, "CollectCSGOExperience"), scenarioId(scenarioId) { };

        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    struct CollectDevExperienceCommand : Node {
        int scenarioId;
        CollectDevExperienceCommand(Blackboard & blackboard, int scenarioId) :
            Node(blackboard, "CollectDevExperience"), scenarioId(scenarioId) { };

        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    struct CollectBotRankingCommand : Node {
        int scenarioId;
        vector<BotType> botScenarioOrder;

        CollectBotRankingCommand(Blackboard & blackboard, int scenarioId, const vector<BotType> & botScenarioOrder) :
            Node(blackboard, "CollectBotRankingCommand"), scenarioId(scenarioId), botScenarioOrder(botScenarioOrder) { };

        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    struct SetUseLearnedModel : Node {
        bool useLearnedModel;
        TeamId teamId;

        SetUseLearnedModel(Blackboard & blackboard, bool useLearnedModel, TeamId teamId) :
                Node(blackboard, "SetUseLearnedModel"), useLearnedModel(useLearnedModel), teamId(teamId) { }

        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    struct SetUseUncertainModel : Node {
        bool useUncertainModel;

        SetUseUncertainModel(Blackboard & blackboard, bool useUncertainModel) :
                Node(blackboard, "SetUseUncertainModel"), useUncertainModel(useUncertainModel) { }

        virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override;
    };

    vector<Script::Ptr> createSurveyScripts(const csknow::plant_states::PlantStatesResult & plantStatesResult,
                                            int startSituationId, bool quitAtEnd, int numHumans);
}


#endif //CSKNOW_TEST_SURVEY_H
