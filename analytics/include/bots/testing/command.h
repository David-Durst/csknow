//
// Created by steam on 6/17/22.
//

#ifndef CSKNOW_COMMAND_H
#define CSKNOW_COMMAND_H

#include "load_save_bot_data.h"
#include <memory>

struct Command {
    using Ptr = std::unique_ptr<Command>;
    virtual ~Command() { };
    virtual string ToString() const = 0;

    template <typename ...Args>
    static vector<Command::Ptr> makeList(Args ...args)
    {
        vector<Command::Ptr> commands;
        constexpr size_t n = sizeof...(Args);
        commands.reserve(n);

        (
                commands.emplace_back(std::move(args)), ...
        );

        return commands;
    }
};

struct InitTestingRound : Command {
    virtual string ToString() const override {
        return "sm_botDebug t; mp_freezetime 0; mp_warmup_end; sm_draw;";
    }
};

struct SavePos : Command {
    string playerName;

    SavePos(CSGOId playerId, const ServerState & serverState)
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

    Teleport(CSGOId playerId, const ServerState & serverState)
        : playerName(serverState.getClient(playerId).name) { }
    Teleport(string playerName) : playerName(playerName) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << "sm_teleport " << playerName;
        return result.str();
    }
};

struct SlayAllBut : Command {
    vector<string> playerNames;

    SlayAllBut(vector<CSGOId> playerIds, const ServerState & serverState) {
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

    SetArmor(CSGOId playerId, const ServerState & serverState, int armorValue)
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

    SetHealth(CSGOId playerId, const ServerState & serverState, int healthValue)
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

    GiveItem(CSGOId playerId, const ServerState & serverState, string itemName)
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

    SetCurrentItem(CSGOId playerId, const ServerState & serverState, string itemName)
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

    SpecPlayerToTarget(CSGOId playerId, CSGOId targetId, const ServerState & serverState, bool thirdPerson = false)
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

    SpecPlayerThirdPerson(CSGOId playerId, const ServerState & serverState)
            : playerName(serverState.getClient(playerId).name) { }
    SpecPlayerThirdPerson(string playerName) : playerName(playerName) { }

    virtual string ToString() const override {
        std::stringstream result;
        result << "sm_specPlayerThirdPerson " << playerName;
        return result.str();
    }
};

#endif //CSKNOW_COMMAND_H
