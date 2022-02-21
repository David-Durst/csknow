//
// Created by durst on 2/20/22.
//
#include <algorithm>
#include "bots/manage_thinkers.h"
using namespace std::chrono_literals;

std::vector<Skill> botSkills {{0.001, true, MovementPolicy::PushOnly}, {7.5, false, MovementPolicy::PushAndRetreat}};
bool firstGame = true;
std::filesystem::directory_entry curDemoFile;
int32_t curMapNumber = -1;
std::map<string, Skill> botNameToSkill;
std::random_device rd;  // Will be used to obtain a seed for the random number engine
std::mt19937 accuracyGen(rd()), otherSkillGen(rd()); // Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> accuracyDis(0., 10.), otherSkillDis(0., 1.);

std::filesystem::directory_entry getLatestDemoFile(string mapsPath) {
    bool firstFile = true;
    std::filesystem::directory_entry mostRecentEntry;

    for (auto &entry : std::filesystem::directory_iterator(mapsPath + "/../")) {
        if (entry.is_regular_file() && entry.path().has_extension() && entry.path().extension() == ".dem") {
            if (firstFile || mostRecentEntry.last_write_time() < entry.last_write_time()) {
                mostRecentEntry = entry;
                firstFile = false;
            }
        }
    }

    return mostRecentEntry;
}

void savePlayersFile() {
    std::filesystem::path p = curDemoFile.path();
    std::ofstream fsPlayers(p.replace_extension(".csv"));
    p += "_skills";
    fsPlayers << "name,maxInaccuracy,stopToShoot,movementPolicy,demo_file\n";
    for (const auto & [name, skill] : botNameToSkill) {
        fsPlayers << name << "," << skill.maxInaccuracy << "," << (skill.stopToShoot ? 1 : 0)
            << "," << enumAsInt(skill.movementPolicy) << "," << curDemoFile.path().filename() << "\n";
    }
    fsPlayers.close();

}

bool haveLatestDemoFile(string mapsPath) {
    return getLatestDemoFile(mapsPath) == curDemoFile.path();
}

void updateThinkers(ServerState & state, string mapsPath, std::list<Thinker> & thinkers) {
    string navPath = mapsPath + "/" + state.mapName + ".nav";
    bool newBotsForThisGame = false;

    // on each new game, clear all the thinkers out
    if (state.mapNumber != curMapNumber) {
        curMapNumber = state.mapNumber;
        thinkers.clear();
    }

    std::map<int32_t, bool> botCSGOIdsToAdd;
    std::map<int32_t, string> botCSGOIdsToNames;
    for (const auto & client : state.clients) {
        if (client.isBot) {
            botCSGOIdsToAdd.insert({client.csgoId, true});
            botCSGOIdsToNames.insert({client.csgoId, client.name});
        }
    }

    // remove all elements that aren't currently bots
    thinkers.remove_if([&](const Thinker & thinker){
        return botCSGOIdsToAdd.find(thinker.getCurBotCSGOId()) == botCSGOIdsToAdd.end() ;
    });

    // mark already controlled bots as not needed to be added
    for (const auto & thinker : thinkers) {
        botCSGOIdsToAdd[thinker.getCurBotCSGOId()] = false;
    }

    // add all uncontrolled bots
    bool firstAdd = true;
    for (const auto & [csgoId, toAdd] : botCSGOIdsToAdd) {
        if (toAdd) {
            // if we are adding any bots, update the demo file list
            if (firstAdd) {
                std::filesystem::directory_entry latestDemoFile = getLatestDemoFile(mapsPath);
                if (firstGame || latestDemoFile.path() != curDemoFile.path()) {
                    curDemoFile = latestDemoFile;
                    botNameToSkill.clear();
                }
                firstGame = false;
                firstAdd = false;
            }

            Skill skill;
            string botName = botCSGOIdsToNames[csgoId];
            // if don't have a skill for this bot, generate one
            if (botNameToSkill.find(botName) == botNameToSkill.end()) {
                bool stopToShoot = otherSkillDis(otherSkillGen) < 0.5;
                MovementPolicy policy = otherSkillDis(otherSkillGen) < 0.5 ?
                                        MovementPolicy::PushAndRetreat : MovementPolicy::PushOnly;
                botNameToSkill.insert({botName, {accuracyDis(accuracyGen), stopToShoot, policy} });
                newBotsForThisGame = true;
            }
            skill = botNameToSkill[botName];
            thinkers.emplace_back(state, csgoId, navPath, skill);
        }
    }
    
    if (newBotsForThisGame) {
        savePlayersFile();
    }

    // clear all old inputs
    // and force existing players to put a new one in
    // this will happen infrequrntly, so unlikely to cause issue with frequent missed inputs
    for (size_t i = 0; i < state.inputsValid.size(); i++) {
        state.inputsValid[i] = false;
    }
}
