//
// Created by steam on 6/17/22.
//

#ifndef CSKNOW_COMMAND_H
#define CSKNOW_COMMAND_H

#include "bots/load_save_bot_data.h"
#include "bots/testing/script_data.h"
#include <memory>

struct Command : Node {
    vector<string> scriptLines;

    Command(Blackboard & blackboard, string name) :
        Node(blackboard, name) { };

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        if (state.saveScript(scriptLines)) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }
        return playerNodeState[treeThinker.csgoId];
    }
};

struct PreTestingInit : Command {
    int numHumansNonSpec;
    PreTestingInit(Blackboard & blackboard, int numHumansNonSpec) :
            Command(blackboard, "PreTestingInitmd"), numHumansNonSpec(numHumansNonSpec) { }
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.inTest = true;
        scriptLines = {"sm_allHumansSpec " + std::to_string(numHumansNonSpec) + "; sm_botDebug f; sm_skipFirstRound;"};
        return Command::exec(state, treeThinker);
    }
};

struct Draw : Command {
    Draw(Blackboard & blackboard) :
            Command(blackboard, "Draw") { }
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.inTest = true;
        scriptLines = {"sm_draw;"};
        return Command::exec(state, treeThinker);
    }
};

struct InitTestingRound : Command {
    string scriptName;
    InitTestingRound(Blackboard & blackboard, string scriptName) :
        Command(blackboard, "InitTestingRoundCmd"), scriptName(scriptName) { }
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.inTest = true;
        scriptLines = {"sm_refresh;say Running Test " + scriptName + "; sm_skipFirstRound; sm_botDebug t; mp_warmup_end; sm_draw;"};
        return Command::exec(state, treeThinker);
    }
};

struct InitGameRound : Command {
    string scriptName;
    InitGameRound(Blackboard & blackboard, string scriptName) :
        Command(blackboard, "InitGameRoundCmd"), scriptName(scriptName) { }
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.inTest = true;
        scriptLines = {"sm_refresh;say Running Game " + scriptName + "; sm_skipFirstRound; sm_botDebug f; mp_warmup_end; sm_draw;"};
        return Command::exec(state, treeThinker);
    }
};

struct SavePos : Command {
    string playerName;

    SavePos(Blackboard & blackboard, CSGOId playerId, const ServerState & serverState) :
        Command(blackboard, "SavePosCmd"), playerName(serverState.getClient(playerId).name) { }
    SavePos(Blackboard & blackboard, string playerName) :
        Command(blackboard, "SavePosCmd"), playerName(playerName) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_savePos " << playerName;
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct SetPos : Command {
    Vec3 pos;
    Vec2 angle;
    SetPos(Blackboard & blackboard, Vec3 pos) :
        Command(blackboard, "SetPosCmd"), pos(pos), angle{0., 0.} { }
    SetPos(Blackboard & blackboard, Vec3 pos, Vec2 angle) :
        Command(blackboard, "SetPosCmd"), pos(pos), angle(angle) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_setPos " << pos.x << " " << pos.y << " " << pos.z << " " << angle.y << " " << angle.x;
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct Teleport : Command {
    string playerName;

    Teleport(Blackboard & blackboard, CSGOId playerId, const ServerState & serverState) :
        Command(blackboard, "TeleportCmd"), playerName(serverState.getClient(playerId).name) { }
    Teleport(Blackboard & blackboard, string playerName) :
        Command(blackboard, "TeleportCmd"), playerName(playerName) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_teleport " << playerName;
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct TeleportPlantedC4 : Command {
    TeleportPlantedC4(Blackboard & blackboard) :
            Command(blackboard, "TeleportPlantedC4Cmd") { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_teleportPlantedC4";
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct SlayAllBut : Command {
    vector<string> playerNames;

    SlayAllBut(Blackboard & blackboard, vector<CSGOId> playerIds, const ServerState & serverState) :
        Command(blackboard, "SlayAllButCmd") {
        for (const auto & playerId : playerIds) {
            playerNames.push_back(serverState.getClient(playerId).name);
        }
    }
    SlayAllBut(Blackboard & blackboard, vector<string> playerNames) :
        Command(blackboard, "SlayAllButCmd"), playerNames(playerNames) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_slayAllBut";
        for (const auto & playerName : playerNames) {
            result << " " << playerName;
        }
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct SetArmor : Command {
    string playerName;
    int armorValue;

    SetArmor(Blackboard & blackboard, CSGOId playerId, const ServerState & serverState, int armorValue) :
        Command(blackboard, "SetArmorCmd"), playerName(serverState.getClient(playerId).name), armorValue(armorValue) { }
    SetArmor(Blackboard & blackboard, string playerName, int armorValue) :
        Command(blackboard, "SetArmorCmd"), playerName(playerName), armorValue(armorValue) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_setArmor " << playerName << " " << armorValue;
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct SetHealth : Command {
    string playerName;
    int healthValue;

    SetHealth(Blackboard & blackboard, CSGOId playerId, const ServerState & serverState, int healthValue) :
        Command(blackboard, "SetHealthCmd"), playerName(serverState.getClient(playerId).name), healthValue(healthValue) { }
    SetHealth(Blackboard & blackboard, string playerName, int healthValue) :
        Command(blackboard, "SetHealthCmd"), playerName(playerName), healthValue(healthValue) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_setHealth " << playerName << " " << healthValue;
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct DamageActive : Command {
    string attackerName, victimName;

    DamageActive(Blackboard & blackboard, CSGOId attackerId, CSGOId victimId, const ServerState & serverState) :
        Command(blackboard, "DamageActiveCmd"), attackerName(serverState.getClient(attackerId).name),
        victimName(serverState.getClient(victimId).name) { }
    DamageActive(Blackboard & blackboard, string attackerName, string victimName) :
        Command(blackboard, "DamageActiveCmd"), attackerName(attackerName), victimName(victimName) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_damageActive " << attackerName << " " << victimName;
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct GiveItem : Command {
    string playerName;
    string itemName;

    GiveItem(Blackboard & blackboard, CSGOId playerId, const ServerState & serverState, string itemName) :
        Command(blackboard, "GiveItem"), playerName(serverState.getClient(playerId).name), itemName(itemName) { }
    GiveItem(Blackboard & blackboard, string playerName, string itemName) :
        Command(blackboard, "GiveItem"), playerName(playerName), itemName(itemName) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_giveItem " << playerName << " " << itemName;
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct RemoveGuns : Command {
    string playerName;

    RemoveGuns(Blackboard & blackboard, CSGOId playerId, const ServerState & serverState) :
            Command(blackboard, "GiveItem"), playerName(serverState.getClient(playerId).name) { }
    RemoveGuns(Blackboard & blackboard, string playerName) :
            Command(blackboard, "GiveItem"), playerName(playerName) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_removeGuns " << playerName;
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct SetCurrentItem : Command {
    string playerName;
    string itemName;

    SetCurrentItem(Blackboard & blackboard, CSGOId playerId, const ServerState & serverState, string itemName) :
        Command(blackboard, "SetCurrentItem"), playerName(serverState.getClient(playerId).name), itemName(itemName) { }
    SetCurrentItem(Blackboard & blackboard, string playerName, string itemName) :
        Command(blackboard, "SetCurrentItem"), playerName(playerName), itemName(itemName) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_setCurrentItem " << playerName << " " << itemName;
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct SpecPlayerToTarget : Command {
    string playerName;
    string targetName;
    bool thirdPerson;

    SpecPlayerToTarget(Blackboard & blackboard, CSGOId playerId, CSGOId targetId, const ServerState & serverState, bool thirdPerson = false) :
        Command(blackboard, "SpecPlayerToTargetCmd"),
        playerName(serverState.getClient(playerId).name), targetName(serverState.getClient(targetId).name), thirdPerson(thirdPerson) { }
    SpecPlayerToTarget(Blackboard & blackboard, string playerName, string targetName, bool thirdPerson = false) :
        Command(blackboard, "SpecPlayerToTargetCmd"),
        playerName(playerName), targetName(targetName), thirdPerson(thirdPerson) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_specPlayerToTarget " << playerName << " " << targetName;
        if (thirdPerson) {
            result << " t";
        }
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct SpecGoto : Command {
    string playerName;
    Vec3 pos;
    Vec2 angle;

    SpecGoto(Blackboard & blackboard, CSGOId playerId, const ServerState & serverState, Vec3 pos, Vec2 angle) :
        Command(blackboard, "SpecGoto"),
        playerName(serverState.getClient(playerId).name), pos(pos), angle(angle) { }
    SpecGoto(Blackboard & blackboard, string playerName, Vec3 pos, Vec2 angle) :
        Command(blackboard, "SpecGoto"),
        playerName(playerName), pos(pos), angle(angle) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "sm_specGoto " << playerName << " " << pos.x << " " << pos.y << " " << pos.z << " " << angle.y << " " << angle.x;
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct SpecDynamic : Command {
    vector<NeededBot> neededBots;
    ObserveSettings observeSettings;
    SpecDynamic(Blackboard & blackboard, vector<NeededBot> neededBots, ObserveSettings observeSettings) :
        Command(blackboard, "SpecDynamic"), neededBots(neededBots), observeSettings(observeSettings) { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        for (const auto & client : state.clients) {
            if (!client.isBot) {
                if (observeSettings.observeType == ObserveType::FirstPerson) {
                    CSGOId neededBotCSGOId = neededBots[observeSettings.neededBotIndex].id;
                    result << "sm_specPlayerToTarget " << client.name << " " << state.getClient(neededBotCSGOId).name << "; ";
                }
                else if (observeSettings.observeType == ObserveType::ThirdPerson) {
                    CSGOId neededBotCSGOId = neededBots[observeSettings.neededBotIndex].id;
                    result << "sm_specPlayerToTarget " << client.name << " " << state.getClient(neededBotCSGOId).name << "t; ";
                }
                else if (observeSettings.observeType == ObserveType::Absolute) {
                    Vec3 pos = observeSettings.cameraOrigin;
                    Vec2 angle = observeSettings.cameraAngle;
                    result << "sm_specGoto " << client.name << " " << pos.x << " " << pos.y << " " << pos.z << " " << angle.x << " " << angle.y << "; ";
                }
            }
        }
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

struct SayIf : Command {
    bool condition;
    string str;
    SayIf(Blackboard & blackboard, bool condition, string str) :
        Command(blackboard, "InitTestingRoundCmd"), condition(condition), str(str) { }
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.inTest = true;
        if (condition) {
            scriptLines = {"say " + str + ";"};
        }
        else {
            scriptLines = {};
        }
        return Command::exec(state, treeThinker);
    }
};

struct Pause : Command {
    Pause(Blackboard & blackboard) : Command(blackboard, "Pause") { }

    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        std::stringstream result;
        result << "pause";
        scriptLines = {result.str()};
        return Command::exec(state, treeThinker);
    }
};

#endif //CSKNOW_COMMAND_H
