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

std::filesystem::directory_entry getLatestDemoFile(string mapsPath);
void savePlayersFile();
void updateThinkers(ServerState & state, string mapsPath, std::list<Thinker> & thinkers, bool useLearned);

#endif //CSKNOW_MANAGE_THINKERS_H
