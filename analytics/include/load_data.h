#ifndef CSKNOW_LOAD_DATA_H
#define CSKNOW_LOAD_DATA_H
#define NUM_PLAYERS 10
#include <string>
#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include "IntervalTree.h"
#include "queries/parser_constants.h"
#include <highfive/H5File.hpp>
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::vector;
using std::pair;
#define ENGINE_TEAM_UNASSIGNED 0
#define ENGINE_TEAM_SPEC 1
#define ENGINE_TEAM_T 2
#define ENGINE_TEAM_CT 3
#define INVALID_ID (-1)
#define WEAPON_RECOIL_SCALE 2.0
#define VIEW_RECOIL_TRACKING 0.45
constexpr char TEAM_UNASSIGNED_NAME[] = "unassigned";
constexpr char TEAM_SPEC_NAME[] = "spectator";
constexpr char TEAM_T_NAME[] = "T";
constexpr char TEAM_CT_NAME[] = "CT";

struct RangeIndexEntry {
    int64_t minId, maxId;

    [[nodiscard]]
    string toCSV() const {
        return std::to_string(minId) + "," + std::to_string(maxId);
    }
};

typedef vector<RangeIndexEntry> RangeIndex;
//typedef IntervalTree<int64_t, int64_t> IntervalIndex;
struct IntervalIndex {
    IntervalTree<int64_t, int64_t> intervalToEvent;
    unordered_map<int64_t, RangeIndexEntry> eventToInterval;
};

class ColStore {
public:
    int64_t size = INVALID_ID;
    vector<string> fileNames;
    vector<int64_t> gameStarts;
    vector<int64_t> id;
    string hdf5Prefix = "/colStore/";
    ColStore(string hdf5Prefix) : hdf5Prefix(hdf5Prefix) { }
    virtual void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) {
        size = rows;
        fileNames.resize(numFiles);
        this->gameStarts = std::move(gameStarts);
        this->id.resize(rows);
    }
    virtual void toHDF5Inner(HighFive::File &, HighFive::DataSetCreateProps &) {
        throw std::runtime_error("HDFS saving not implemented for this query yet");
    }
    void toHDF5(HighFive::File & file);
    virtual void fromHDF5(HighFive::File &) {
        throw std::runtime_error("HDFS loading not implemented for this query yet");
    }
};

class Equipment : public ColStore {
public:
    vector<string> name;

    Equipment() : ColStore("/equipment/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        name.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Equipment & lhs, const Equipment & rhs);

class GameTypes : public ColStore {
public:
    vector<string> tableType;

    GameTypes() : ColStore("/game types/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        tableType.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const GameTypes & lhs, const GameTypes & rhs);

class HitGroups : public ColStore {
public:
    vector<string> groupName;

    HitGroups() : ColStore("/hit groups/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        groupName.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const HitGroups & lhs, const HitGroups & rhs);

class Games : public ColStore {
public:
    vector<string> demoFile;
    vector<double> demoTickRate;
    vector<double> gameTickRate;
    vector<string> mapName;
    vector<int64_t> gameType;
    RangeIndex roundsPerGame;
    RangeIndex playersPerGame;

    Games() : ColStore("/games/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        demoFile.resize(rows);
        demoTickRate.resize(rows);
        gameTickRate.resize(rows);
        mapName.resize(rows);
        gameType.resize(rows);
        roundsPerGame.resize(rows);
        playersPerGame.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Games & lhs, const Games & rhs);

class Players : public ColStore {
public:
    vector<int64_t> gameId;
    vector<string> name;
    vector<int64_t> steamId;
    // add this offset to id to get the row entry
    int64_t idOffset = 1;

    Players() : ColStore("/players/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        gameId.resize(rows);
        name.resize(rows);
        steamId.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Players & lhs, const Players & rhs);

class Rounds : public ColStore {
public:
    vector<int64_t> gameId;
    vector<int64_t> startTick;
    // tick when objective accomplished
    vector<int64_t> endTick;
    // tick when run around time ends
    vector<int64_t> endOfficialTick;
    vector<bool> warmup;
    vector<bool> overtime;
    vector<int64_t> freezeTimeEnd;
    vector<int16_t> roundNumber;
    vector<int16_t> roundEndReason;
    vector<int16_t> winner;
    vector<int16_t> tWins;
    vector<int16_t> ctWins;
    RangeIndex ticksPerRound;
    // add this offset to id to get the row entry
    int64_t idOffset = 1;

    Rounds() : ColStore("/rounds/") { }
    Rounds(const string & hdf5Prefix) : ColStore(hdf5Prefix) { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        gameId.resize(rows);
        startTick.resize(rows);
        endTick.resize(rows);
        endOfficialTick.resize(rows);
        warmup.resize(rows);
        overtime.resize(rows);
        freezeTimeEnd.resize(rows);
        roundNumber.resize(rows);
        roundEndReason.resize(rows);
        winner.resize(rows);
        tWins.resize(rows);
        ctWins.resize(rows);
        ticksPerRound.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Rounds & lhs, const Rounds & rhs);

class Ticks: public ColStore {
public:
    vector<int64_t> roundId;
    vector<int64_t> gameTime;
    vector<int64_t> demoTickNumber;
    vector<int64_t> gameTickNumber;
    vector<int64_t> bombCarrier;
    vector<double> bombX;
    vector<double> bombY;
    vector<double> bombZ;
    RangeIndex patPerTick;
    RangeIndex spottedPerTick;
    RangeIndex footstepPerTick;
    IntervalIndex weaponFirePerTick;
    IntervalIndex killsPerTick;
    IntervalIndex hurtPerTick;
    IntervalIndex grenadesPerTick;
    IntervalIndex grenadesThrowPerTick;
    IntervalIndex grenadesActivePerTick;
    IntervalIndex grenadesExpiredPerTick;
    IntervalIndex grenadesDestroyedPerTick;
    IntervalIndex flashedPerTick;
    IntervalIndex plantsPerTick;
    IntervalIndex plantsStartPerTick;
    IntervalIndex plantsEndPerTick;
    IntervalIndex defusalsPerTick;
    IntervalIndex defusalsStartPerTick;
    IntervalIndex defusalsEndPerTick;
    IntervalIndex explosionsPerTick;
    IntervalIndex sayPerTick;

    Ticks() : ColStore("/ticks/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        roundId.resize(rows);
        gameTime.resize(rows);
        demoTickNumber.resize(rows);
        gameTickNumber.resize(rows);
        bombCarrier.resize(rows);
        bombX.resize(rows);
        bombY.resize(rows);
        bombZ.resize(rows);
        patPerTick.resize(rows);
        spottedPerTick.resize(rows);
        footstepPerTick.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Ticks & lhs, const Ticks & rhs);

class PlayerAtTick: public ColStore {
public:
    vector<int64_t> playerId;
    vector<int64_t> tickId;
    vector<double> posX;
    vector<double> posY;
    vector<double> posZ;
    vector<double> eyePosZ;
    vector<double> velX;
    vector<double> velY;
    vector<double> velZ;
    vector<double> viewX;
    vector<double> viewY;
    vector<double> aimPunchX;
    vector<double> aimPunchY;
    vector<double> viewPunchX;
    vector<double> viewPunchY;
    vector<double> recoilIndex;
    vector<double> nextPrimaryAttack;
    vector<double> nextSecondaryAttack;
    vector<double> gameTime;
    vector<int16_t> team;
    vector<double> health;
    vector<double> armor;
    vector<bool> hasHelmet;
    vector<bool> isAlive;
    vector<bool> duckingKeyPressed;
    vector<double> duckAmount;
    vector<bool> isReloading;
    vector<bool> isWalking;
    vector<bool> isScoped;
    vector<bool> isAirborne;
    vector<double> flashDuration;
    vector<int16_t> activeWeapon;
    vector<int16_t> primaryWeapon;
    vector<int16_t> primaryBulletsClip;
    vector<int16_t> primaryBulletsReserve;
    vector<int16_t> secondaryWeapon;
    vector<int16_t> secondaryBulletsClip;
    vector<int16_t> secondaryBulletsReserve;
    vector<int16_t> numHe;
    vector<int16_t> numFlash;
    vector<int16_t> numSmoke;
    vector<int16_t> numMolotov;
    vector<int16_t> numIncendiary;
    vector<int16_t> numDecoy;
    vector<int16_t> numZeus;
    vector<bool> hasDefuser;
    vector<bool> hasBomb;
    vector<int32_t> money;
    vector<int32_t> ping;

    PlayerAtTick() : ColStore("/player at tick/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        playerId.resize(rows);
        tickId.resize(rows);
        posX.resize(rows);
        posY.resize(rows);
        posZ.resize(rows);
        eyePosZ.resize(rows);
        velX.resize(rows);
        velY.resize(rows);
        velZ.resize(rows);
        viewX.resize(rows);
        viewY.resize(rows);
        aimPunchX.resize(rows);
        aimPunchY.resize(rows);
        viewPunchX.resize(rows);
        viewPunchY.resize(rows);
        recoilIndex.resize(rows);
        nextPrimaryAttack.resize(rows);
        nextSecondaryAttack.resize(rows);
        gameTime.resize(rows);
        team.resize(rows);
        health.resize(rows);
        armor.resize(rows);
        hasHelmet.resize(rows);
        isAlive.resize(rows);
        duckingKeyPressed.resize(rows);
        duckAmount.resize(rows);
        isReloading.resize(rows);
        isWalking.resize(rows);
        isScoped.resize(rows);
        isAirborne.resize(rows);
        flashDuration.resize(rows);
        activeWeapon.resize(rows);
        primaryWeapon.resize(rows);
        primaryBulletsClip.resize(rows);
        primaryBulletsReserve.resize(rows);
        secondaryWeapon.resize(rows);
        secondaryBulletsClip.resize(rows);
        secondaryBulletsReserve.resize(rows);
        numHe.resize(rows);
        numFlash.resize(rows);
        numSmoke.resize(rows);
        numMolotov.resize(rows);
        numIncendiary.resize(rows);
        numDecoy.resize(rows);
        numZeus.resize(rows);
        hasDefuser.resize(rows);
        hasBomb.resize(rows);
        money.resize(rows);
        ping.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;

    void makePitchNeg90To90() {
#pragma omp parallel for
        for (int64_t i = 0; i < size; i++) {
            if (viewY[i] > 260.0) {
                viewY[i] -= 360;
            }
        }
    }
};
bool operator==(const PlayerAtTick & lhs, const PlayerAtTick & rhs);

class Spotted : public ColStore {
public:
    vector<int64_t> tickId;
    vector<int64_t> spottedPlayer;
    vector<int64_t> spotterPlayer;
    vector<bool> isSpotted;

    Spotted() : ColStore("/spotted/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        tickId.resize(rows);
        spottedPlayer.resize(rows);
        spotterPlayer.resize(rows);
        isSpotted.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Spotted & lhs, const Spotted & rhs);

class Footstep : public ColStore {
public:
    vector<int64_t> tickId;
    vector<int64_t> steppingPlayer;

    Footstep() : ColStore("/footstep/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        tickId.resize(rows);
        steppingPlayer.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Footstep & lhs, const Footstep & rhs);

class WeaponFire : public ColStore {
public:
    vector<int64_t> tickId;
    vector<int64_t> shooter;
    vector<int16_t> weapon;

    WeaponFire() : ColStore("/weapon fire/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        tickId.resize(rows);
        shooter.resize(rows);
        weapon.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const WeaponFire & lhs, const WeaponFire & rhs);

class Kills : public ColStore {
public:
    vector<int64_t> tickId;
    vector<int64_t> killer;
    vector<int64_t> victim;
    vector<int16_t> weapon;
    vector<int64_t> assister;
    vector<bool> isHeadshot;
    vector<bool> isWallbang;
    vector<int32_t> penetratedObjects;

    Kills() : ColStore("/kills/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        tickId.resize(rows);
        killer.resize(rows);
        victim.resize(rows);
        weapon.resize(rows);
        assister.resize(rows);
        isHeadshot.resize(rows);
        isWallbang.resize(rows);
        penetratedObjects.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Kills & lhs, const Kills & rhs);

class Hurt : public ColStore {
public:
    vector<int64_t> tickId;
    vector<int64_t> victim;
    vector<int64_t> attacker;
    vector<DemoEquipmentType> weapon;
    vector<int32_t> armorDamage;
    vector<int32_t> armor;
    vector<int32_t> healthDamage;
    vector<int32_t> health;
    vector<int64_t> hitGroup;

    Hurt() : ColStore("/hurt/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        tickId.resize(rows);
        victim.resize(rows);
        attacker.resize(rows);
        weapon.resize(rows);
        armorDamage.resize(rows);
        armor.resize(rows);
        healthDamage.resize(rows);
        health.resize(rows);
        hitGroup.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Hurt & lhs, const Hurt & rhs);

class Grenades : public ColStore {
public:
    vector<int64_t> thrower;
    vector<int16_t> grenadeType;
    vector<int64_t> throwTick;
    vector<int64_t> activeTick;
    vector<int64_t> expiredTick;
    vector<int64_t> destroyTick;
    RangeIndex flashedPerGrenade;
    IntervalIndex trajectoryPerGrenade;

    Grenades() : ColStore("/grenades/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        thrower.resize(rows);
        grenadeType.resize(rows);
        throwTick.resize(rows);
        activeTick.resize(rows);
        expiredTick.resize(rows);
        destroyTick.resize(rows);
        flashedPerGrenade.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Grenades & lhs, const Grenades & rhs);

class Flashed : public ColStore {
public:
    vector<int64_t> tickId;
    vector<int64_t> grenadeId;
    vector<int64_t> thrower;
    vector<int64_t> victim;

    Flashed() : ColStore("/flashed/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        tickId.resize(rows);
        grenadeId.resize(rows);
        thrower.resize(rows);
        victim.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Flashed & lhs, const Flashed & rhs);

class GrenadeTrajectories : public ColStore {
public:
    vector<int64_t> grenadeId;
    vector<int32_t> idPerGrenade;
    vector<double> posX;
    vector<double> posY;
    vector<double> posZ;

    GrenadeTrajectories() : ColStore("/grenade trajectories/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        grenadeId.resize(rows);
        idPerGrenade.resize(rows);
        posX.resize(rows);
        posY.resize(rows);
        posZ.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const GrenadeTrajectories & lhs, const GrenadeTrajectories & rhs);

class Plants : public ColStore {
public:
    vector<int64_t> startTick;
    vector<int64_t> endTick;
    vector<int64_t> planter;
    vector<bool> succesful;
    RangeIndex defusalsPerPlant;
    RangeIndex explosionsPerPlant;

    Plants() : ColStore("/plants/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        startTick.resize(rows);
        endTick.resize(rows);
        planter.resize(rows);
        succesful.resize(rows);
        defusalsPerPlant.resize(rows);
        explosionsPerPlant.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Plants & lhs, const Plants & rhs);

class Defusals : public ColStore {
public:
    vector<int64_t> plantId;
    vector<int64_t> startTick;
    vector<int64_t> endTick;
    vector<int64_t> defuser;
    vector<bool> succesful;

    Defusals() : ColStore("/defusals/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        plantId.resize(rows);
        startTick.resize(rows);
        endTick.resize(rows);
        defuser.resize(rows);
        succesful.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Defusals & lhs, const Defusals & rhs);

class Explosions : public ColStore {
public:
    vector<int64_t> plantId;
    vector<int64_t> tickId;

    Explosions() : ColStore("/explosions/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        plantId.resize(rows);
        tickId.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Explosions & lhs, const Explosions & rhs);

class Say : public ColStore {
public:
    vector<int64_t> gameId;
    vector<int64_t> tickId;
    vector<string> message;

    Say() : ColStore("/say/") { }
    void init(int64_t rows, int64_t numFiles, vector<int64_t> gameStarts) override {
        ColStore::init(rows, numFiles, gameStarts);
        gameId.resize(rows);
        tickId.resize(rows);
        message.resize(rows);
    }
    void toHDF5Inner(HighFive::File & file, HighFive::DataSetCreateProps & hdf5FlatCreateProps) override;
    void fromHDF5(HighFive::File & file) override;
};
bool operator==(const Say & lhs, const Say & rhs);

void loadDataCSV(Equipment & equipment, GameTypes & gameTypes, HitGroups & hitGroups, Games & games, Players & players,
                 Rounds & unfilteredRounds, Rounds & filteredRounds, Ticks & ticks, PlayerAtTick & playerAtTick, Spotted & spotted, Footstep & footstep, WeaponFire & weaponFire,
                 Kills & kills, Hurt & hurt, Grenades & grenades, Flashed & flashed, GrenadeTrajectories & grenadeTrajectories,
                 Plants & plants, Defusals & defusals, Explosions & explosions, Say & say, const string & dataPath);

void loadDataHDF5(Equipment & equipment, GameTypes & gameTypes, HitGroups & hitGroups, Games & games, Players & players,
                  Rounds & unfilteredRounds, Rounds & filteredRounds, Ticks & ticks, PlayerAtTick & playerAtTick, Spotted & spotted, Footstep & footstep, WeaponFire & weaponFire,
                  Kills & kills, Hurt & hurt, Grenades & grenades, Flashed & flashed, GrenadeTrajectories & grenadeTrajectories,
                  Plants & plants, Defusals & defusals, Explosions & explosions, Say & say, const string & dataPath);

void saveDataHDF5(Equipment & equipment, GameTypes & gameTypes, HitGroups & hitGroups, Games & games, Players & players,
                  Rounds & unfilteredRounds, Rounds & filteredRounds, Ticks & ticks, PlayerAtTick & playerAtTick, Spotted & spotted, Footstep & footstep, WeaponFire & weaponFire,
                  Kills & kills, Hurt & hurt, Grenades & grenades, Flashed & flashed, GrenadeTrajectories & grenadeTrajectories,
                  Plants & plants, Defusals & defusals, Explosions & explosions, Say & say, const string & filePath);

#endif //CSKNOW_LOAD_DATA_H
