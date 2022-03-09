//
// Created by durst on 2/20/22.
//
#include "bots/manage_thinkers.h"
#include <algorithm>
#include <file_helpers.h>

using namespace std::chrono_literals;

Skill terminatorSkill = {false, 0.001, true, MovementPolicy::PushOnly};
Skill badAggressiveSkill = {false, 10.0, false, MovementPolicy::PushOnly};
Skill badStopSkill = {false, 10.0, true, MovementPolicy::PushOnly};
Skill afraidSkill = {false, 10.0, false, MovementPolicy::PushAndRetreat};
Skill nothingSkill = {false, 10.0, true, MovementPolicy::HoldOnly};
std::vector<Skill> allBotSkills {terminatorSkill, badAggressiveSkill, badStopSkill, afraidSkill, nothingSkill};
std::vector<Skill> teamBotSkills {terminatorSkill, nothingSkill};
std::filesystem::directory_entry curDemoFile;
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
                Skill newBotSkill;
                if (curMode == BotAddMode::PickFromSkillVector) {
                    newBotSkill = allBotSkills[botNameToSkill.size() % allBotSkills.size()];
                }
                else if (curMode == BotAddMode::TerminatorCTStoppedT) {
                    int32_t skillIndex = botCSGOIdsToTeam[csgoId] == 3 ? 0 : 1;
                    newBotSkill = teamBotSkills[skillIndex];
                }
                // this is BotAddMode::GenRandom
                else {
                    bool stopToShoot = otherSkillDis(otherSkillGen) < 0.5;
                    MovementPolicy policy = otherSkillDis(otherSkillGen) < 0.5 ?
                                            MovementPolicy::PushAndRetreat : MovementPolicy::PushOnly;
                    newBotSkill = {false, accuracyDis(accuracyGen), stopToShoot, policy};
                }

                // if learned, find most similar in list of players and make parameters actually that
                if (useLearned) {
                    newBotSkill = findMostSimilarSkillFromTraining(newBotSkill);
                }

                botNameToSkill.insert({botName, newBotSkill});
                newBotsForThisGame = true;
            }
            skill = botNameToSkill[botName];
            thinkers.emplace_back(state, csgoId, navPath, skill, pythonPlanState);
        }
    }
    
    if (newBotsForThisGame) {
        saveSkillsDuringPlay();
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

void ManageThinkerState::saveSkillsDuringPlay() {
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

void ManageThinkerState::saveSkillsDuringTraining(string outputPath) {
    std::ofstream fsSkills(outputPath);
    fsSkills << "player_id,learned,maxInaccuracy,stopToShoot,movementPolicy\n";
    for (size_t i = 0; i < skillsFromFile.size(); i++) {
        fsSkills << i << "," << (skillsFromFile[i].learned ? 1 : 0) << ","
                 << skillsFromFile[i].maxInaccuracy << ","
                 << (skillsFromFile[i].stopToShoot ? 1 : 0) << ","
                 << enumAsInt(skillsFromFile[i].movementPolicy) << "\n";
    }
    fsSkills.close();
}

void ManageThinkerState::loadSkillsDuringTraining(const Games & games, const Players & players) {
    string skillFileName = "skill.csv";
    string skillFilePath = dataPath + "/" + skillFileName;

    map<string, map<string, int64_t>> demoNameToPlayerNameToPlayerId;
    for (int64_t gameIndex = 0; gameIndex < games.size; gameIndex++) {
        for (int64_t playerIndex = games.playersPerGame[gameIndex].minId;
                playerIndex <= games.playersPerGame[gameIndex].maxId; playerIndex++) {
            string demoFileName = games.demoFile[gameIndex];
            // remove machine id from demo file name for now as player's don't know it. will need to add it later
            string demoSuffix = "Global_Offensive";
            size_t suffixStart = demoFileName.find(demoSuffix);
            string subsetDemoFileName = demoFileName.substr(0, suffixStart + demoSuffix.size()) + ".dem";
            demoNameToPlayerNameToPlayerId[subsetDemoFileName][players.name[playerIndex]] = players.id[playerIndex];
        }
    }

    // load skills into here, will reorder by player index later
    map<int64_t, Skill> playerIdToSkill;

    // mmap the file
    auto [fd, stats, file] = openMMapFile(skillFilePath);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = 0;

    Skill tmpSkill;
    int32_t tmpMovementPolicy;
    string tmpPlayerName, tmpDemoName;

    for (size_t curStart = 0, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, tmpPlayerName);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, tmpSkill.learned);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, tmpSkill.maxInaccuracy);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, tmpSkill.stopToShoot);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, tmpMovementPolicy);
            tmpSkill.movementPolicy = intAsEnum<MovementPolicy>(tmpMovementPolicy);
        }
        else if (colNumber == 5) {
            readCol(file, curStart, curDelimiter, tmpDemoName);
            tmpDemoName.erase(remove(tmpDemoName.begin(), tmpDemoName.end(), '\"'), tmpDemoName.end());
            assert(demoNameToPlayerNameToPlayerId.find(tmpDemoName) != demoNameToPlayerNameToPlayerId.end());
            assert(demoNameToPlayerNameToPlayerId[tmpDemoName].find(tmpPlayerName) !=
                    demoNameToPlayerNameToPlayerId[tmpDemoName].end());
            playerIdToSkill[demoNameToPlayerNameToPlayerId[tmpDemoName][tmpPlayerName]] = tmpSkill;
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 6;
    }
    closeMMapFile({fd, stats, file});

    for (const auto & [playerId, skill] : playerIdToSkill) {
        assert(playerId == skillsFromFile.size());
        skillsFromFile.push_back(skill);
    }
    assert(players.size == skillsFromFile.size());
}

void ManageThinkerState::loadSkillsAfterTraining() {
    string skillFileName = "train_skills.csv";
    string skillFilePath = pythonDataPath + "/" + skillFileName;

    // mmap the file
    auto [fd, stats, file] = openMMapFile(skillFilePath);

    // track location for error logging
    int64_t rowNumber = 0;
    int64_t colNumber = 0;

    // track location for insertion
    int64_t arrayEntry = 0;

    Skill tmpSkill;
    int32_t tmpMovementPolicy;
    string tmpPlayerName, tmpDemoName;

    for (size_t curStart = 0, curDelimiter = getNextDelimiter(file, curStart, stats.st_size);
         curDelimiter < stats.st_size;
         curStart = curDelimiter + 1, curDelimiter = getNextDelimiter(file, curStart, stats.st_size)) {
        if (colNumber == 0) {
            readCol(file, curStart, curDelimiter, tmpPlayerName);
        }
        else if (colNumber == 1) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, tmpSkill.learned);
        }
        else if (colNumber == 2) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, tmpSkill.maxInaccuracy);
        }
        else if (colNumber == 3) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, tmpSkill.stopToShoot);
        }
        else if (colNumber == 4) {
            readCol(file, curStart, curDelimiter, rowNumber, colNumber, tmpMovementPolicy);
            tmpSkill.movementPolicy = intAsEnum<MovementPolicy>(tmpMovementPolicy);
            skillsFromFile.push_back(tmpSkill);
            rowNumber++;
            arrayEntry++;
        }
        colNumber = (colNumber + 1) % 5;
    }
    closeMMapFile({fd, stats, file});
}

Skill ManageThinkerState::findMostSimilarSkillFromTraining(Skill inputSkill) {
    if (skillsFromFile.empty()) {
        loadSkillsAfterTraining();
    }

    double minInaccuracyDifference = std::numeric_limits<double>::max();
    Skill resultSkill;

    for (const auto & candidateSkill : skillsFromFile) {
        if (candidateSkill.stopToShoot == inputSkill.stopToShoot &&
            candidateSkill.movementPolicy == inputSkill.movementPolicy) {
            double newInaccuracyDifference = std::abs(candidateSkill.maxInaccuracy - inputSkill.maxInaccuracy);
            if (newInaccuracyDifference < minInaccuracyDifference) {
                minInaccuracyDifference = newInaccuracyDifference;
                resultSkill = candidateSkill;
            }

        }
    }

    // if nothing matches, just take the first one
    if (minInaccuracyDifference == std::numeric_limits<double>::max()) {
        resultSkill = skillsFromFile[0];
    }
    resultSkill.learned = true;
    return resultSkill;
}

