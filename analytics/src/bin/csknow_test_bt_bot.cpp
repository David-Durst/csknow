#include "bots/load_save_bot_data.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/script.h"
#include "bots/testing/scripts/navigation/test_entry.h"
#include "bots/testing/scripts/navigation/test_hold.h"
#include "bots/testing/scripts/test_aim.h"
#include "bots/testing/scripts/test_memory.h"
#include "bots/testing/scripts/test_communication.h"
#include "bots/testing/scripts/test_teamwork.h"
#include "bots/testing/scripts/test_danger.h"
#include "bots/testing/scripts/test_possible_nav_areas.h"
#include "bots/testing/scripts/test_engage_spacing.h"
#include "bots/testing/scripts/test_defuse.h"
#include "bots/testing/scripts/test_head_position.h"
#include "bots/testing/scripts/test_round.h"
#include "queries/moments/plant_states.h"
#include "bots/analysis/learned_models.h"
#include "navmesh/nav_file.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <list>
#include <map>
#include <filesystem>
//#define LOG_STATE

int main(int argc, char * argv[]) {
    if (argc != 8) {
        std::cout << "please call this code with 7 arguments: \n"
            << "1. path/to/maps\n"
            << "2. path/to/data\n"
            << "3. path/to/log\n"
            << "4. path/to/models\n"
            << "5. path/to/saved/data\n"
            << "6. t for tests, r for rounds, rh for rounds with hueristics, rht for rounds with t hueristics, rhct for rounds with ct heuristics\n"
            << "7. 1 for all csknow bots, ct for ct only csknow bots, t for t only csknow bots, 0 for for no csknow bots\n"
            << std::endl;
        return 1;
    }
    string mapsPath = argv[1], dataPath = argv[2], logPath = argv[3], modelsDir = argv[4], savedDatasetsDir = argv[5],
        roundsTestStr = argv[6], botStop = argv[7];

    bool runTest = roundsTestStr == "t";
    if (!runTest) {
        processModelArg(roundsTestStr);
    }

    ServerState state;
    state.dataPath = dataPath;

    uint64_t numFailures = 0;
    Tree tree(modelsDir);
    std::thread filterReceiver(&Tree::readFilterNames, &tree);

    SetBotStop setBotStop(*tree.blackboard, botStop);
    setBotStop.exec(state, tree.defaultThinker);
    SetMaxRounds setMaxRounds(*tree.blackboard, 100, false);
    setMaxRounds.exec(state, tree.defaultThinker);

    bool finishedTests = false;
    csknow::plant_states::PlantStatesResult plantStatesResult;
    plantStatesResult.load(savedDatasetsDir + "/plantStates.hdf5");
    ScriptsRunner roundScriptsRunner(createRoundScripts(plantStatesResult, false), true);

    ScriptsRunner scriptsRunner(Script::makeList(
            /*
         make_unique<GooseToCatScript>(state),
            make_unique<GooseToCatShortScript>(state),
            make_unique<CTPushLongScript>(state),
            make_unique<CTPushBDoorsScript>(state),
            make_unique<CTPushBHoleScript>(state),
            make_unique<DefuseScript>(state),
            make_unique<HoldLongScript>(state),
            make_unique<HoldASitePushScript>(state),
            make_unique<HoldASiteBaitScript>(state),
            make_unique<HoldBSitePushScript>(state),
            make_unique<HoldBSiteBaitScript>(state),
             */
            make_unique<AimAndKillWithinTimeCheck>(state)
             /*
            make_unique<CTEngageSpacingScript>(state),
            make_unique<PushBaitGooseToCatScript>(state),
            make_unique<PushWaitForBaitGooseToCatScript>(state),
            make_unique<PushMultipleBaitGooseToCatScript>(state),
            make_unique<PushLurkBaitASiteScript>(state),
            make_unique<PushATwoOrdersScript>(state),
            make_unique<PushTwoBDoorsScript>(state),
            make_unique<PushThreeBScript>(state),
            make_unique<MemoryAimCheck>(state),
            make_unique<MemoryForgetCheck>(state),
            make_unique<CommunicationAimCheck>(state),
            make_unique<CommunicationForgetCheck>(state),
            make_unique<CommunicationIgnoreCheck>(state),
            make_unique<SpawnPossibleNavAreasCheck>(state),
            make_unique<DiffusionPossibleNavAreasCheck>(state),
            make_unique<VisibilityPossibleNavAreasCheck>(state),
            make_unique<DangerOnePlayerCheck>(state),
            make_unique<DangerTwoPlayerCheck>(state)
                 */
    ), true);
    ScriptsRunner scenarioRunner(Script::makeList(
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, false),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, false)
    ), false, 0);

    ScriptsRunner humanScenarioRunner(Script::makeList(
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::Close, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::BottomRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::None,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Forward,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Left,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::HardLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidLeft, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightUp, true),
        make_unique<variable_aim_test::VariableAimAndKillWithinTimeCheck>(
            variable_aim_test::EnemyPos::TopRamp, variable_aim_test::EnemyMovement::Right,
            variable_aim_test::AttackerInitialViewAngle::MidRightDown, true)
    ), false, 1);
     /*
    // visualization scripts rather than actual tests
    ScriptsRunner scriptsRunner(Script::makeList(
            make_unique<HeadTrackingScript>(state),
            make_unique<CrouchedHeadTrackingScript>(state)
    ), true);
     */


    at::set_num_threads(1);

    int32_t priorFrame = 0;
    size_t numMisses = 0;
    size_t numSkips = 0;
    size_t numDups = 0;
    auto priorStart = std::chrono::system_clock::now();
    CSGOFileTime priorFileTime;
    [[maybe_unused]] double priorGameTime = 0;
    [[maybe_unused]] double priorStatTime = 0;
#pragma clang diagnostic push
#pragma ide diagnostic ignored "EndlessLoop"
    while (!finishedTests) {
        auto start = std::chrono::system_clock::now();
        double curStatTime = state.getGeneralStatFileTime();
        CSGOFileTime newFileTime = state.loadServerState();
        double newGameTime = state.gameTime;
        std::chrono::duration<double> timePerTick(state.tickInterval);
        auto parseEnd = std::chrono::system_clock::now();
        std::chrono::duration<double> startToStart = start - priorStart;
        std::chrono::duration<double> fileWriteToWrite = newFileTime - priorFileTime;
        double frameDiff = (state.getLastFrame() - priorFrame) / 128.;

        if (state.loadedSuccessfully) {
            tree.tick(state, mapsPath);
            if (state.clients.size() > 0) {
                //std::cout << "time since last save " << state.getSecondsBetweenTimes(start, priorStart) << std::endl;
                if (runTest) {
                    scriptsRunner.initialize(tree, state);
                    finishedTests = scriptsRunner.tick(tree, state);
                    //scenarioRunner.initialize(tree, state);
                    //finishedTests = scenarioRunner.tick(tree, state);
                    //humanScenarioRunner.initialize(tree, state);
                    //finishedTests = humanScenarioRunner.tick(tree, state);
                }
                else {
                    roundScriptsRunner.initialize(tree, state);
                    finishedTests = roundScriptsRunner.tick(tree, state);
                }
            }
            state.saveBotInputs();
        }
        else {
            numFailures++;
        }

        /*
        if (state.getLastFrame() != priorFrame + 1) {
            std::cout << "write to write: " << fileWriteToWrite.count()
                      << ", delta game time " << newGameTime - priorGameTime
                      << ", delta stat time" << curStatTime - priorStatTime
                      << ", cur stat time" << curStatTime << std::endl;
            numSkips++;
        }
         */
        if (state.getLastFrame() == priorFrame) {
            numDups++;
        }
        priorStart = start;
        priorGameTime = newGameTime;
        priorStatTime = curStatTime;
        priorFileTime = newFileTime;
        priorFrame = state.getLastFrame();

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> botTime = end - start;
        std::chrono::duration<double> parseTime = parseEnd - start;

        std::fstream logFile (logPath + "/bt_bot.log", std::fstream::out);
        std::fstream testLogFile (logPath + "/bt_test_bot.log", std::fstream::out);
        logFile << "Num failures " << numFailures << ", last bad path: " << state.badPath << std::endl;
        if (botTime < timePerTick) {
            logFile << "Bot compute time: ";
        }
        else {
            logFile << "\033[1;31mMissed Bot compute time:\033[0m " ;
            numMisses++;
        }
        logFile << botTime.count() << "s, pct parse " << parseTime.count() / botTime.count()
            << ", start to start " <<  startToStart.count()
            << ", frame to time ratio " << frameDiff / startToStart.count()
            << ", num misses " << numMisses
            << ", num skips " << numSkips
            << ", num dups " << numDups << std::endl;
        logFile << tree.curLog;
        logFile.close();
        if (runTest) {
            testLogFile << scriptsRunner.curLog();
        }
        else {
            testLogFile << roundScriptsRunner.curLog();
        }
        testLogFile.close();
        state.sleepUntilServerStateExists(priorFileTime);
    }
#pragma clang diagnostic pop

    exit(0);
}
