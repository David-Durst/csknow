//
// Created by durst on 2/20/22.
//
#include "bots/manage_thinkers.h"
#include <algorithm>
using namespace std::chrono_literals;

Skill learnedTerminatorSkill = {true, 0.001, true, MovementPolicy::PushOnly};
Skill terminatorSkill = {false, 0.001, true, MovementPolicy::PushOnly};
Skill badAggressiveSkill = {false, 10.0, false, MovementPolicy::PushOnly};
Skill badStopSkill = {false, 10.0, true, MovementPolicy::PushOnly};
Skill afraidSkill = {false, 10.0, false, MovementPolicy::PushAndRetreat};
Skill nothingSkill = {false, 10.0, true, MovementPolicy::HoldOnly};
std::vector<Skill> allBotSkills {terminatorSkill, badAggressiveSkill, badStopSkill, afraidSkill, nothingSkill};
std::vector<Skill> teamBotSkills {terminatorSkill, nothingSkill};
std::vector<Skill> learnedBotSkills {learnedTerminatorSkill, nothingSkill};
bool pickFromSkillVector = false, pickByTeam = false;
bool firstGame = true;
std::filesystem::directory_entry curDemoFile;
int32_t curMapNumber = -1;
std::map<string, Skill> botNameToSkill;
std::random_device rd;  // Will be used to obtain a seed for the random number engine
std::mt19937 accuracyGen(rd()), otherSkillGen(rd()); // Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> accuracyDis(0., 10.), otherSkillDis(0., 1.);

std::filesystem::directory_entry
ManageThinkerState::getLatestDemoFile(string mapsPath) {
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

void ManageThinkerState::savePlayersFile() {
    std::filesystem::path p = curDemoFile.path();
    std::ofstream fsPlayers(p.replace_extension(".csv"));
    p += "_skills";
    fsPlayers << "name,learned,maxInaccuracy,stopToShoot,movementPolicy,demo_file\n";
    for (const auto & [name, skill] : botNameToSkill) {
        fsPlayers << name << "," << (skill.learned ? 1 : 0) << "," << skill.maxInaccuracy << "," << (skill.stopToShoot ? 1 : 0)
            << "," << enumAsInt(skill.movementPolicy) << "," << curDemoFile.path().filename() << "\n";
    }
    fsPlayers.close();

}

void
ManageThinkerState::updateThinkers(ServerState & state, string mapsPath, std::list<Thinker> & thinkers, bool useLearned) {
    string navPath = mapsPath + "/" + state.mapName + ".nav";
    bool newBotsForThisGame = false;

    // on each new game, clear all the thinkers out and update nav file for python interconnect
    if (state.mapNumber != curMapNumber) {
        curMapNumber = state.mapNumber;
        thinkers.clear();
        pythonPlanState.reloadNavFile(navPath);
    }

    std::map<int32_t, bool> botCSGOIdsToAdd;
    std::map<int32_t, string> botCSGOIdsToNames;
    std::map<int32_t, int32_t> botCSGOIdsToTeam;
    for (const auto & client : state.clients) {
        if (client.isBot) {
            botCSGOIdsToAdd.insert({client.csgoId, true});
            botCSGOIdsToNames.insert({client.csgoId, client.name});
            botCSGOIdsToTeam.insert({client.csgoId, client.team});
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
                if (pickFromSkillVector) {
                    botNameToSkill.insert({botName, allBotSkills[botNameToSkill.size() % allBotSkills.size()]});
                }
                else if (pickByTeam) {
                    int32_t skillIndex = botCSGOIdsToTeam[csgoId] == 3 ? 0 : 1;
                    if (useLearned) {
                        botNameToSkill.insert({botName, learnedBotSkills[skillIndex]});
                    }
                    else {
                        botNameToSkill.insert({botName, teamBotSkills[skillIndex]});
                    }
                }
                else {
                    bool stopToShoot = otherSkillDis(otherSkillGen) < 0.5;
                    MovementPolicy policy = otherSkillDis(otherSkillGen) < 0.5 ?
                                            MovementPolicy::PushAndRetreat : MovementPolicy::PushOnly;
                    botNameToSkill.insert({botName, {false, accuracyDis(accuracyGen), stopToShoot, policy} });
                }
                newBotsForThisGame = true;
            }
            skill = botNameToSkill[botName];
            thinkers.emplace_back(state, csgoId, navPath, skill, pythonPlanState);
        }
    }
    
    if (newBotsForThisGame) {
        savePlayersFile();
    }

    if (useLearned) {
        pythonPlanState.CommunicateWithPython();
    }

    // clear all old inputs
    // and force existing players to put a new one in
    // this will happen infrequrntly, so unlikely to cause issue with frequent missed inputs
    for (size_t i = 0; i < state.inputsValid.size(); i++) {
        state.inputsValid[i] = false;
    }
}
