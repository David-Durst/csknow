//
// Created by steam on 6/17/22.
//

#ifndef CSKNOW_COMMAND_H
#define CSKNOW_COMMAND_H

#include "load_save_bot_data.h"

struct Command {
    virtual string ToString() const = 0;
};

struct SavePos : Command {
    string playerName;

    SavePos(CSKnowId playerId, ServerState serverState)
            : playerName(serverState.getClient(playerId).name) { }
    SavePos(string playerName) : playerName(playerName) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << "sm_savePos " << playerName;
        return result.str();
    }
};

struct SetPos : Command {
    Vec3 pos;
    Vec2 angle;
    SetPos(Vec3 pos) : pos(pos), angle{0., 0.} { }
    SetPos(Vec3 pos, Vec2 angle) : pos(pos), angle(angle) { }

    virtual string ToString() const override {
        std::stringstream result;
        // flipped because of yaw/pitch flipping (see load_save_bot_data.h)
        result << "sm_setPos " << pos.x << " " << pos.y << " " << pos.z << " " << angle.y << " " << angle.x;
        return result.str();
    }
};

struct Teleport : Command {
    string playerName;

    Teleport(CSKnowId playerId, ServerState serverState)
        : playerName(serverState.getClient(playerId).name) { }
    Teleport(string playerName) : playerName(playerName) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << "sm_teleport " << playerName;
        return result.str();
    }
};

struct TeleportToPos : SetPos, Teleport {
    TeleportToPos(Vec3 pos, Vec2 angle, CSKnowId playerId, ServerState serverState)
            : SetPos(pos, angle), Teleport(playerId, serverState) { }
    TeleportToPos(Vec3 pos, Vec2 angle, string playerName)
            : SetPos(pos, angle), Teleport(playerName) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << SetPos::ToString();
        result << Teleport::ToString();
        return result.str();
    }
};

struct SlayAllBut : Command {
    vector<string> playerNames;

    SlayAllBut(vector<CSKnowId> playerIds, ServerState serverState) {
        for (const auto & playerId : playerIds) {
            playerNames.push_back(serverState.getClient(playerId).name);
        }
    }
    SlayAllBut(vector<string> playerNames) : playerNames(playerNames) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << "sm_slayAllBut";
        for (const auto & playerName : playerNames) {
            result << " " << playerName;
        }
        return result.str();
    }
};

struct SetArmor : Command {
    string playerName;
    int armorValue;

    SetArmor(CSKnowId playerId, ServerState serverState, int armorValue)
            : playerName(serverState.getClient(playerId).name), armorValue(armorValue) { }
    SetArmor(string playerName, int armorValue) : playerName(playerName), armorValue(armorValue) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << "sm_setArmor " << playerName << " " << armorValue;
        return result.str();
    }
};

struct SetHealth : Command {
    string playerName;
    int healthValue;

    SetHealth(CSKnowId playerId, ServerState serverState, int healthValue)
            : playerName(serverState.getClient(playerId).name), healthValue(healthValue) { }
    SetHealth(string playerName, int healthValue) : playerName(playerName), healthValue(healthValue) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << "sm_setHealth " << playerName << " " << healthValue;
        return result.str();
    }
};

struct GiveItem : Command {
    string playerName;
    string itemName;

    GiveItem(CSKnowId playerId, ServerState serverState, string itemName)
            : playerName(serverState.getClient(playerId).name), itemName(itemName) { }
    GiveItem(string playerName, string itemName) : playerName(playerName), itemName(itemName) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << "sm_giveItem " << playerName << " " << itemName;
        return result.str();
    }
};

struct SetCurrentItem : Command {
    string playerName;
    string itemName;

    SetCurrentItem(CSKnowId playerId, ServerState serverState, string itemName)
            : playerName(serverState.getClient(playerId).name), itemName(itemName) { }
    SetCurrentItem(string playerName, string itemName) : playerName(playerName), itemName(itemName) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << "sm_setCurrentItem " << playerName << " " << itemName;
        return result.str();
    }
};

struct SpecPlayerToTarget : Command {
    string playerName;
    string targetName;
    bool thirdPerson;

    SpecPlayerToTarget(CSKnowId playerId, CSKnowId targetId, ServerState serverState, bool thirdPerson = false)
            : playerName(serverState.getClient(playerId).name), targetName(serverState.getClient(targetId).name), thirdPerson(thirdPerson) { }
    SpecPlayerToTarget(string playerName, string targetName, bool thirdPerson = false) : playerName(playerName), targetName(targetName), thirdPerson(thirdPerson) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << "sm_setCurrentItem " << playerName << " " << targetName;
        if (thirdPerson) {
            result << " t";
        }
        return result.str();
    }
};

struct SpecPlayerThirdPerson : Command {
    string playerName;

    SpecPlayerThirdPerson(CSKnowId playerId, ServerState serverState)
            : playerName(serverState.getClient(playerId).name) { }
    SpecPlayerThirdPerson(string playerName, string itemName) : playerName(playerName) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << "sm_specPlayerThirdPerson " << playerName;
        return result.str();
    }
};

#endif //CSKNOW_COMMAND_H
