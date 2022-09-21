//
// Created by durst on 2/20/22.
//
#ifdef false
#include "load_save_bot_data.h"
#include "bots/thinker.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <list>
#include <map>
#include <filesystem>

#ifndef CSKNOW_MANAGE_THINKERS_H
#define CSKNOW_MANAGE_THINKERS_H

// need to move more of the state at top of manage_thinkers.cpp to here
struct ManageThinkerState {
    PythonModelInterface pythonPlanState;
    string dataPath, pythonDataPath;
    vector<Skill> skillsFromFile;

    bool firstGame = true;
    int32_t curMapNumber = -1;

    enum class BotAddMode {
        PickFromSkillVector,
        TerminatorCTStoppedT,
        GenRandom,
        NUM_ADD_MODES
    };
    BotAddMode curMode = BotAddMode::GenRandom;

    std::filesystem::directory_entry getLatestDemoFile(string mapsPath);
    void updateThinkers(ServerState & state, string mapsPath, std::list<Thinker> & thinkers, bool useLearned);
    void saveSkillsDuringPlay();
    void saveSkillsDuringTraining(string outputPath);
    void loadSkillsDuringTraining(const Games & games, const Players & players);
    void loadSkillsAfterTraining();
    Skill findMostSimilarSkillFromTraining(Skill inputSkill);

    ManageThinkerState(string dataPath, string pythonDataPath)
        : pythonPlanState(dataPath), dataPath(dataPath), pythonDataPath(pythonDataPath) { };
};

#endif //CSKNOW_MANAGE_THINKERS_H

#endif //false