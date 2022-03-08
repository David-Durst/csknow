//
// Created by durst on 2/20/22.
//
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
    string dataPath;
    vector<Skill> skillsFromFile;

    std::filesystem::directory_entry getLatestDemoFile(string mapsPath);
    void updateThinkers(ServerState & state, string mapsPath, std::list<Thinker> & thinkers, bool useLearned);
    void saveSkillsDuringPlay();
    void saveSkillsDuringTraining(string outputPath);
    void loadSkills(const Games & games, const Players & players);

    ManageThinkerState(string dataPath) : pythonPlanState(dataPath), dataPath(dataPath) { };
};

#endif //CSKNOW_MANAGE_THINKERS_H
