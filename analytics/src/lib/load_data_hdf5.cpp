//
// Created by durst on 5/16/23.
//

#include "load_data.h"
#include "hdf5_helpers.h"

void ColStore::toHDF5(const std::string &filePath) {
    // We create an empty HDF55 file, by truncating an existing
    // file if required:
    HighFive::File file(filePath, HighFive::File::Overwrite);

    HighFive::DataSetCreateProps hdf5CreateProps;
    hdf5CreateProps.add(HighFive::Deflate(6));
    hdf5CreateProps.add(HighFive::Chunking(id.size()));
    file.createDataSet(hdf5Prefix + "id", id, hdf5CreateProps);

    // create all other columns
    toHDF5Inner(file);
}

void Equipment::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "name", name, hdf5FlatCreateProps);
}

void Equipment::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    name = file.getDataSet(hdf5Prefix + "name").read<std::vector<string>>();
    size = static_cast<int64_t>(id.size());
}

void GameTypes::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "table type", tableType, hdf5FlatCreateProps);
}

void GameTypes::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    tableType = file.getDataSet(hdf5Prefix + "table type").read<std::vector<string>>();
    size = static_cast<int64_t>(id.size());
}

void HitGroups::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "group name", groupName, hdf5FlatCreateProps);
}

void HitGroups::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    groupName = file.getDataSet(hdf5Prefix + "group name").read<std::vector<string>>();
    size = static_cast<int64_t>(id.size());
}

void Games::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "demo file", demoFile, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "demo tick rate", demoTickRate, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "game tick rate", gameTickRate, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "map name", mapName, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "game type", gameType, hdf5FlatCreateProps);
}

void Games::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    demoFile = file.getDataSet(hdf5Prefix + "demo file").read<std::vector<string>>();
    demoTickRate = file.getDataSet(hdf5Prefix + "demo tick rate").read<std::vector<double>>();
    gameTickRate = file.getDataSet(hdf5Prefix + "game tick rate").read<std::vector<double>>();
    mapName = file.getDataSet(hdf5Prefix + "map name").read<std::vector<string>>();
    gameType = file.getDataSet(hdf5Prefix + "game type").read<std::vector<int64_t>>();
    size = static_cast<int64_t>(id.size());
}

void Players::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "game id", gameId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "name", name, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "steam id", steamId, hdf5FlatCreateProps);
}

void Players::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    gameId = file.getDataSet(hdf5Prefix + "game id").read<std::vector<int64_t>>();
    name = file.getDataSet(hdf5Prefix + "name").read<std::vector<string>>();
    steamId = file.getDataSet(hdf5Prefix + "steam id").read<std::vector<int64_t>>();
    size = static_cast<int64_t>(id.size());
}

void Rounds::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "game id", gameId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "start tick", startTick, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "end tick", endTick, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "end official tick", endOfficialTick, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "warmup", warmup, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "overtime", overtime, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "freeze time end", freezeTimeEnd, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "round number", roundNumber, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "round end reason", roundEndReason, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "winner", winner, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "t wins", tWins, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "ct wins", ctWins, hdf5FlatCreateProps);
}

void Rounds::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    gameId = file.getDataSet(hdf5Prefix + "game id").read<std::vector<int64_t>>();
    startTick = file.getDataSet(hdf5Prefix + "start tick").read<std::vector<int64_t>>();
    endTick = file.getDataSet(hdf5Prefix + "end tick").read<std::vector<int64_t>>();
    endOfficialTick = file.getDataSet(hdf5Prefix + "end official tick").read<std::vector<int64_t>>();
    warmup = file.getDataSet(hdf5Prefix + "warmup").read<std::vector<bool>>();
    overtime = file.getDataSet(hdf5Prefix + "overtime").read<std::vector<bool>>();
    freezeTimeEnd = file.getDataSet(hdf5Prefix + "freeze time end").read<std::vector<int64_t>>();
    roundNumber = file.getDataSet(hdf5Prefix + "round number").read<std::vector<int16_t>>();
    roundEndReason = file.getDataSet(hdf5Prefix + "round end reason").read<std::vector<int16_t>>();
    winner = file.getDataSet(hdf5Prefix + "winner").read<std::vector<int16_t>>();
    tWins = file.getDataSet(hdf5Prefix + "t wins").read<std::vector<int16_t>>();
    ctWins = file.getDataSet(hdf5Prefix + "ct wins").read<std::vector<int16_t>>();
    size = static_cast<int64_t>(id.size());
}

void Ticks::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "round id", roundId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "game time", gameTime, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "demo tick number", demoTickNumber, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "game tick number", gameTickNumber, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "bomb carrier", bombCarrier, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "bomb x", bombX, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "bomb y", bombY, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "bomb z", bombZ, hdf5FlatCreateProps);
}

void Ticks::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    roundId = file.getDataSet(hdf5Prefix + "round id").read<std::vector<int64_t>>();
    gameTime = file.getDataSet(hdf5Prefix + "game time").read<std::vector<int64_t>>();
    demoTickNumber = file.getDataSet(hdf5Prefix + "demo tick number").read<std::vector<int64_t>>();
    gameTickNumber = file.getDataSet(hdf5Prefix + "game tick number").read<std::vector<int64_t>>();
    bombCarrier = file.getDataSet(hdf5Prefix + "bomb carrier").read<std::vector<int64_t>>();
    bombX = file.getDataSet(hdf5Prefix + "bomb x").read<std::vector<double>>();
    bombY = file.getDataSet(hdf5Prefix + "bomb y").read<std::vector<double>>();
    bombZ = file.getDataSet(hdf5Prefix + "bomb z").read<std::vector<double>>();
    size = static_cast<int64_t>(id.size());
}

void PlayerAtTick::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "player id", playerId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "pos x", posX, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "pos y", posY, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "pos z", posZ, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "eye pos z", eyePosZ, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "vel x", velX, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "vel y", velY, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "vel z", velZ, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "view x", viewX, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "view y", viewY, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "aim punch x", aimPunchX, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "aim punch y", aimPunchY, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "view punch x", viewPunchX, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "view punch y", viewPunchY, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "recoil index", recoilIndex, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "next primary attack", nextPrimaryAttack, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "next secondary attack", nextSecondaryAttack, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "game time", gameTime, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "team", team, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "health", health, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "armor", armor, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "has helmet", hasHelmet, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "is alive", isAlive, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "ducking key pressed", duckingKeyPressed, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "duck amount", duckAmount, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "is reloading", isReloading, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "is walking", isWalking, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "is scoped", isScoped, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "is airborne", isAirborne, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "flash duration", flashDuration, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "active weapon", activeWeapon, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "primary weapon", primaryWeapon, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "primary bullets clip", primaryBulletsClip, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "primary bullets reserve", primaryBulletsReserve, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "secondary weapon", primaryWeapon, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "secondary bullets clip", primaryBulletsClip, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "secondary bullets reserve", primaryBulletsReserve, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "num HE", numHe, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "num flash", numFlash, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "num molotov", numMolotov, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "num incendiary", numIncendiary, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "num decoy", numDecoy, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "num zeus", numZeus, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "has defuser", hasDefuser, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "has bomb", hasBomb, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "money", money, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "ping", ping, hdf5FlatCreateProps);
}

void PlayerAtTick::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    playerId = file.getDataSet(hdf5Prefix + "player id").read<std::vector<int64_t>>();
    tickId = file.getDataSet(hdf5Prefix + "tick id").read<std::vector<int64_t>>();
    posX = file.getDataSet(hdf5Prefix + "pos x").read<std::vector<double>>();
    posY = file.getDataSet(hdf5Prefix + "pos y").read<std::vector<double>>();
    posZ = file.getDataSet(hdf5Prefix + "pos z").read<std::vector<double>>();
    eyePosZ = file.getDataSet(hdf5Prefix + "eye pos z").read<std::vector<double>>();
    velX = file.getDataSet(hdf5Prefix + "vel x").read<std::vector<double>>();
    velY = file.getDataSet(hdf5Prefix + "vel y").read<std::vector<double>>();
    velZ = file.getDataSet(hdf5Prefix + "vel z").read<std::vector<double>>();
    viewX = file.getDataSet(hdf5Prefix + "view x").read<std::vector<double>>();
    viewY = file.getDataSet(hdf5Prefix + "view y").read<std::vector<double>>();
    aimPunchX = file.getDataSet(hdf5Prefix + "aim punch x").read<std::vector<double>>();
    aimPunchY = file.getDataSet(hdf5Prefix + "aim punch y").read<std::vector<double>>();
    viewPunchX = file.getDataSet(hdf5Prefix + "view punch x").read<std::vector<double>>();
    viewPunchY = file.getDataSet(hdf5Prefix + "view punch y").read<std::vector<double>>();
    recoilIndex = file.getDataSet(hdf5Prefix + "recoil index").read<std::vector<double>>();
    nextPrimaryAttack = file.getDataSet(hdf5Prefix + "next primary attack").read<std::vector<double>>();
    nextSecondaryAttack = file.getDataSet(hdf5Prefix + "next secondary attack").read<std::vector<double>>();
    gameTime = file.getDataSet(hdf5Prefix + "game time").read<std::vector<double>>();
    team = file.getDataSet(hdf5Prefix + "team").read<std::vector<int16_t>>();
    health = file.getDataSet(hdf5Prefix + "health").read<std::vector<double>>();
    armor = file.getDataSet(hdf5Prefix + "armor").read<std::vector<double>>();
    hasHelmet = file.getDataSet(hdf5Prefix + "has helmet").read<std::vector<bool>>();
    isAlive = file.getDataSet(hdf5Prefix + "is alive").read<std::vector<bool>>();
    duckingKeyPressed = file.getDataSet(hdf5Prefix + "ducking key pressed").read<std::vector<bool>>();
    duckAmount = file.getDataSet(hdf5Prefix + "duck amount").read<std::vector<double>>();
    isReloading = file.getDataSet(hdf5Prefix + "is reloading").read<std::vector<bool>>();
    isWalking = file.getDataSet(hdf5Prefix + "is walking").read<std::vector<bool>>();
    isScoped = file.getDataSet(hdf5Prefix + "is scoped").read<std::vector<bool>>();
    isAirborne = file.getDataSet(hdf5Prefix + "is airborne").read<std::vector<bool>>();
    flashDuration = file.getDataSet(hdf5Prefix + "flash duration").read<std::vector<double>>();
    activeWeapon = file.getDataSet(hdf5Prefix + "active weapon").read<std::vector<int16_t>>();
    primaryWeapon = file.getDataSet(hdf5Prefix + "primary weapon").read<std::vector<int16_t>>();
    primaryBulletsClip = file.getDataSet(hdf5Prefix + "primary bullets clip").read<std::vector<int16_t>>();
    primaryBulletsReserve = file.getDataSet(hdf5Prefix + "primary bullets reserve").read<std::vector<int16_t>>();
    secondaryWeapon = file.getDataSet(hdf5Prefix + "secondary weapon").read<std::vector<int16_t>>();
    secondaryBulletsClip = file.getDataSet(hdf5Prefix + "secondary bullets clip").read<std::vector<int16_t>>();
    secondaryBulletsReserve = file.getDataSet(hdf5Prefix + "secondary bullets reserve").read<std::vector<int16_t>>();
    numHe = file.getDataSet(hdf5Prefix + "num HE").read<std::vector<int16_t>>();
    numFlash = file.getDataSet(hdf5Prefix + "num flash").read<std::vector<int16_t>>();
    numMolotov = file.getDataSet(hdf5Prefix + "num molotov").read<std::vector<int16_t>>();
    numIncendiary = file.getDataSet(hdf5Prefix + "num incendiary").read<std::vector<int16_t>>();
    numDecoy = file.getDataSet(hdf5Prefix + "num decoy").read<std::vector<int16_t>>();
    numZeus = file.getDataSet(hdf5Prefix + "num zeus").read<std::vector<int16_t>>();
    hasDefuser = file.getDataSet(hdf5Prefix + "has defuser").read<std::vector<bool>>();
    hasBomb = file.getDataSet(hdf5Prefix + "has bomb").read<std::vector<bool>>();
    money = file.getDataSet(hdf5Prefix + "money").read<std::vector<int32_t>>();
    ping = file.getDataSet(hdf5Prefix + "ping").read<std::vector<int32_t>>();
    size = static_cast<int64_t>(id.size());
}

void Spotted::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "spotted player", spottedPlayer, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "spotter player", spotterPlayer, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "is spotted", isSpotted, hdf5FlatCreateProps);
}

void Spotted::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    tickId = file.getDataSet(hdf5Prefix + "tick id").read<std::vector<int64_t>>();
    spottedPlayer = file.getDataSet(hdf5Prefix + "spotted player").read<std::vector<int64_t>>();
    spotterPlayer = file.getDataSet(hdf5Prefix + "spotter player").read<std::vector<int64_t>>();
    isSpotted = file.getDataSet(hdf5Prefix + "is spotted").read<std::vector<bool>>();
    size = static_cast<int64_t>(id.size());
}

void Footstep::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "stepping player", steppingPlayer, hdf5FlatCreateProps);
}

void Footstep::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    tickId = file.getDataSet(hdf5Prefix + "tick id").read<std::vector<int64_t>>();
    steppingPlayer = file.getDataSet(hdf5Prefix + "stepping player").read<std::vector<int64_t>>();
    size = static_cast<int64_t>(id.size());
}

void WeaponFire::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "shooter", shooter, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "weapon", shooter, hdf5FlatCreateProps);
}

void WeaponFire::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    tickId = file.getDataSet(hdf5Prefix + "tick id").read<std::vector<int64_t>>();
    shooter = file.getDataSet(hdf5Prefix + "shooter").read<std::vector<int64_t>>();
    weapon = file.getDataSet(hdf5Prefix + "weapon").read<std::vector<int16_t>>();
    size = static_cast<int64_t>(id.size());
}

void Kills::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "killer", killer, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "victim", victim, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "weapon", weapon, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "assister", assister, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "is headshot", isHeadshot, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "is wallbang", isWallbang, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "penetrated objects", penetratedObjects, hdf5FlatCreateProps);
}

void Kills::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    tickId = file.getDataSet(hdf5Prefix + "tick id").read<std::vector<int64_t>>();
    killer = file.getDataSet(hdf5Prefix + "killer").read<std::vector<int64_t>>();
    weapon = file.getDataSet(hdf5Prefix + "weapon").read<std::vector<int16_t>>();
    assister = file.getDataSet(hdf5Prefix + "assister").read<std::vector<int64_t>>();
    isHeadshot = file.getDataSet(hdf5Prefix + "is headshot").read<std::vector<bool>>();
    isWallbang = file.getDataSet(hdf5Prefix + "is wallbang").read<std::vector<bool>>();
    penetratedObjects = file.getDataSet(hdf5Prefix + "penetrated objects").read<std::vector<int32_t>>();
    size = static_cast<int64_t>(id.size());
}

void Hurt::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "victim", victim, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "attacker", attacker, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "weapon", vectorOfEnumsToVectorOfInts(weapon), hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "armor damage", armorDamage, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "armor", armor, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "healthDamage", healthDamage, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "health", health, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "hit group", hitGroup, hdf5FlatCreateProps);
}

void Hurt::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    tickId = file.getDataSet(hdf5Prefix + "tick id").read<std::vector<int64_t>>();
    victim = file.getDataSet(hdf5Prefix + "victim").read<std::vector<int64_t>>();
    attacker = file.getDataSet(hdf5Prefix + "attacker").read<std::vector<int64_t>>();
    loadVectorOfEnums(file, hdf5Prefix + "weapon", weapon);
    armorDamage = file.getDataSet(hdf5Prefix + "armor damage").read<std::vector<int32_t>>();
    armor = file.getDataSet(hdf5Prefix + "armor").read<std::vector<int32_t>>();
    healthDamage = file.getDataSet(hdf5Prefix + "health damage").read<std::vector<int32_t>>();
    health = file.getDataSet(hdf5Prefix + "health").read<std::vector<int32_t>>();
    hitGroup = file.getDataSet(hdf5Prefix + "hit group").read<std::vector<int64_t>>();
    size = static_cast<int64_t>(id.size());
}

void Grenades::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "thrower", thrower, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "grenade type", grenadeType, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "throw tick", throwTick, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "active tick", activeTick, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "expired tick", expiredTick, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "destroy tick", destroyTick, hdf5FlatCreateProps);
}

void Grenades::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    thrower = file.getDataSet(hdf5Prefix + "thrower").read<std::vector<int64_t>>();
    grenadeType = file.getDataSet(hdf5Prefix + "grenade type").read<std::vector<int16_t>>();
    throwTick = file.getDataSet(hdf5Prefix + "throw tick").read<std::vector<int64_t>>();
    activeTick = file.getDataSet(hdf5Prefix + "active tick").read<std::vector<int64_t>>();
    expiredTick = file.getDataSet(hdf5Prefix + "expired tick").read<std::vector<int64_t>>();
    destroyTick = file.getDataSet(hdf5Prefix + "destroy tick").read<std::vector<int64_t>>();
    size = static_cast<int64_t>(id.size());
}

void Flashed::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "grenade id", grenadeId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "thrower", thrower, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "victim", victim, hdf5FlatCreateProps);
}

void Flashed::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    tickId = file.getDataSet(hdf5Prefix + "tick id").read<std::vector<int64_t>>();
    thrower = file.getDataSet(hdf5Prefix + "thrower").read<std::vector<int64_t>>();
    victim = file.getDataSet(hdf5Prefix + "victim").read<std::vector<int64_t>>();
    size = static_cast<int64_t>(id.size());
}

void GrenadeTrajectories::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "grenade id", grenadeId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "id per grenade", idPerGrenade, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "pos x", posX, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "pos y", posY, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "pos z", posZ, hdf5FlatCreateProps);
}

void GrenadeTrajectories::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    grenadeId = file.getDataSet(hdf5Prefix + "grenade id").read<std::vector<int64_t>>();
    idPerGrenade = file.getDataSet(hdf5Prefix + "id per grenade").read<std::vector<int32_t>>();
    posX = file.getDataSet(hdf5Prefix + "pos x").read<std::vector<double>>();
    posY = file.getDataSet(hdf5Prefix + "pos y").read<std::vector<double>>();
    posZ = file.getDataSet(hdf5Prefix + "pos z").read<std::vector<double>>();
    size = static_cast<int64_t>(id.size());
}

void Plants::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "start tick", startTick, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "end tick", endTick, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "planter", planter, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "succesful", succesful, hdf5FlatCreateProps);
}

void Plants::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    startTick = file.getDataSet(hdf5Prefix + "start tick").read<std::vector<int64_t>>();
    endTick = file.getDataSet(hdf5Prefix + "end tick").read<std::vector<int64_t>>();
    planter = file.getDataSet(hdf5Prefix + "planter").read<std::vector<int64_t>>();
    succesful = file.getDataSet(hdf5Prefix + "succesful").read<std::vector<bool>>();
    size = static_cast<int64_t>(id.size());
}

void Defusals::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "plant id", plantId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "start tick", startTick, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "end tick", endTick, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "defuser", defuser, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "succesful", succesful, hdf5FlatCreateProps);
}

void Defusals::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    startTick = file.getDataSet(hdf5Prefix + "plant id").read<std::vector<int64_t>>();
    startTick = file.getDataSet(hdf5Prefix + "start tick").read<std::vector<int64_t>>();
    endTick = file.getDataSet(hdf5Prefix + "end tick").read<std::vector<int64_t>>();
    defuser = file.getDataSet(hdf5Prefix + "defuser").read<std::vector<int64_t>>();
    succesful = file.getDataSet(hdf5Prefix + "succesful").read<std::vector<bool>>();
    size = static_cast<int64_t>(id.size());
}

void Explosions::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "plant id", plantId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "tick id", tickId, hdf5FlatCreateProps);
}

void Explosions::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    plantId = file.getDataSet(hdf5Prefix + "plant id").read<std::vector<int64_t>>();
    tickId = file.getDataSet(hdf5Prefix + "tick id").read<std::vector<int64_t>>();
    size = static_cast<int64_t>(id.size());
}

void Say::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet(hdf5Prefix + "game id", gameId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet(hdf5Prefix + "message", message, hdf5FlatCreateProps);
}

void Say::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet(hdf5Prefix + "id").read<std::vector<int64_t>>();
    gameId = file.getDataSet(hdf5Prefix + "game id").read<std::vector<int64_t>>();
    tickId = file.getDataSet(hdf5Prefix + "tick id").read<std::vector<int64_t>>();
    message = file.getDataSet(hdf5Prefix + "message").read<std::vector<string>>();
    size = static_cast<int64_t>(id.size());
}
