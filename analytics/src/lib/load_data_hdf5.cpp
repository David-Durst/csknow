//
// Created by durst on 5/16/23.
//

#include "load_data.h"

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

    file.createDataSet("/data/name", name, hdf5FlatCreateProps);
}

void Equipment::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    name = file.getDataSet("/data/name").read<std::vector<string>>();
    size = static_cast<int64_t>(id.size());
}

void GameTypes::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/table type", tableType, hdf5FlatCreateProps);
}

void GameTypes::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    tableType = file.getDataSet("/data/table type").read<std::vector<string>>();
    size = static_cast<int64_t>(id.size());
}

void HitGroups::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/group name", groupName, hdf5FlatCreateProps);
}

void HitGroups::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    groupName = file.getDataSet("/data/group name").read<std::vector<string>>();
    size = static_cast<int64_t>(id.size());
}

void Games::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/demo file", demoFile, hdf5FlatCreateProps);
    file.createDataSet("/data/demo tick rate", demoTickRate, hdf5FlatCreateProps);
    file.createDataSet("/data/game tick rate", gameTickRate, hdf5FlatCreateProps);
    file.createDataSet("/data/map name", mapName, hdf5FlatCreateProps);
    file.createDataSet("/data/game type", gameType, hdf5FlatCreateProps);
}

void Games::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    demoFile = file.getDataSet("/data/demo file").read<std::vector<string>>();
    demoTickRate = file.getDataSet("/data/demo tick rate").read<std::vector<double>>();
    gameTickRate = file.getDataSet("/data/game tick rate").read<std::vector<double>>();
    mapName = file.getDataSet("/data/map name").read<std::vector<string>>();
    gameType = file.getDataSet("/data/game type").read<std::vector<int64_t>>();
    size = static_cast<int64_t>(id.size());
}

void Players::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/game id", gameId, hdf5FlatCreateProps);
    file.createDataSet("/data/name", name, hdf5FlatCreateProps);
    file.createDataSet("/data/steam id", steamId, hdf5FlatCreateProps);
}

void Players::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    gameId = file.getDataSet("/data/game id").read<std::vector<int64_t>>();
    name = file.getDataSet("/data/name").read<std::vector<string>>();
    steamId = file.getDataSet("/data/steam id").read<std::vector<int64_t>>();
    size = static_cast<int64_t>(id.size());
}

void Rounds::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/game id", gameId, hdf5FlatCreateProps);
    file.createDataSet("/data/start tick", startTick, hdf5FlatCreateProps);
    file.createDataSet("/data/end tick", endTick, hdf5FlatCreateProps);
    file.createDataSet("/data/end official tick", endOfficialTick, hdf5FlatCreateProps);
    file.createDataSet("/data/warmup", warmup, hdf5FlatCreateProps);
    file.createDataSet("/data/overtime", overtime, hdf5FlatCreateProps);
    file.createDataSet("/data/freeze time end", freezeTimeEnd, hdf5FlatCreateProps);
    file.createDataSet("/data/round number", roundNumber, hdf5FlatCreateProps);
    file.createDataSet("/data/round end reason", roundEndReason, hdf5FlatCreateProps);
    file.createDataSet("/data/winner", winner, hdf5FlatCreateProps);
    file.createDataSet("/data/t wins", tWins, hdf5FlatCreateProps);
    file.createDataSet("/data/ct wins", ctWins, hdf5FlatCreateProps);
}

void Rounds::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    gameId = file.getDataSet("/data/game id").read<std::vector<int64_t>>();
    startTick = file.getDataSet("/data/start tick").read<std::vector<int64_t>>();
    endTick = file.getDataSet("/data/end tick").read<std::vector<int64_t>>();
    endOfficialTick = file.getDataSet("/data/end official tick").read<std::vector<int64_t>>();
    warmup = file.getDataSet("/data/warmup").read<std::vector<bool>>();
    overtime = file.getDataSet("/data/overtime").read<std::vector<bool>>();
    freezeTimeEnd = file.getDataSet("/data/freeze time end").read<std::vector<int64_t>>();
    roundNumber = file.getDataSet("/data/round number").read<std::vector<int16_t>>();
    roundEndReason = file.getDataSet("/data/round end reason").read<std::vector<int16_t>>();
    winner = file.getDataSet("/data/winner").read<std::vector<int16_t>>();
    tWins = file.getDataSet("/data/t wins").read<std::vector<int16_t>>();
    ctWins = file.getDataSet("/data/ct wins").read<std::vector<int16_t>>();
    size = static_cast<int64_t>(id.size());
}

void Ticks::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/round id", roundId, hdf5FlatCreateProps);
    file.createDataSet("/data/game time", gameTime, hdf5FlatCreateProps);
    file.createDataSet("/data/demo tick number", demoTickNumber, hdf5FlatCreateProps);
    file.createDataSet("/data/game tick number", gameTickNumber, hdf5FlatCreateProps);
    file.createDataSet("/data/bomb carrier", bombCarrier, hdf5FlatCreateProps);
    file.createDataSet("/data/bomb x", bombX, hdf5FlatCreateProps);
    file.createDataSet("/data/bomb y", bombY, hdf5FlatCreateProps);
    file.createDataSet("/data/bomb z", bombZ, hdf5FlatCreateProps);
}

void Ticks::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    roundId = file.getDataSet("/data/round id").read<std::vector<int64_t>>();
    gameTime = file.getDataSet("/data/game time").read<std::vector<int64_t>>();
    demoTickNumber = file.getDataSet("/data/demo tick number").read<std::vector<int64_t>>();
    gameTickNumber = file.getDataSet("/data/game tick number").read<std::vector<int64_t>>();
    bombCarrier = file.getDataSet("/data/bomb carrier").read<std::vector<int64_t>>();
    bombX = file.getDataSet("/data/bomb x").read<std::vector<double>>();
    bombY = file.getDataSet("/data/bomb y").read<std::vector<double>>();
    bombZ = file.getDataSet("/data/bomb z").read<std::vector<double>>();
    size = static_cast<int64_t>(id.size());
}

void PlayerAtTick::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/player id", playerId, hdf5FlatCreateProps);
    file.createDataSet("/data/tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet("/data/pos x", posX, hdf5FlatCreateProps);
    file.createDataSet("/data/pos y", posY, hdf5FlatCreateProps);
    file.createDataSet("/data/pos z", posZ, hdf5FlatCreateProps);
    file.createDataSet("/data/eye pos z", eyePosZ, hdf5FlatCreateProps);
    file.createDataSet("/data/vel x", velX, hdf5FlatCreateProps);
    file.createDataSet("/data/vel y", velY, hdf5FlatCreateProps);
    file.createDataSet("/data/vel z", velZ, hdf5FlatCreateProps);
    file.createDataSet("/data/view x", viewX, hdf5FlatCreateProps);
    file.createDataSet("/data/view y", viewY, hdf5FlatCreateProps);
    file.createDataSet("/data/aim punch x", aimPunchX, hdf5FlatCreateProps);
    file.createDataSet("/data/aim punch y", aimPunchY, hdf5FlatCreateProps);
    file.createDataSet("/data/view punch x", viewPunchX, hdf5FlatCreateProps);
    file.createDataSet("/data/view punch y", viewPunchY, hdf5FlatCreateProps);
    file.createDataSet("/data/recoil index", recoilIndex, hdf5FlatCreateProps);
    file.createDataSet("/data/next primary attack", nextPrimaryAttack, hdf5FlatCreateProps);
    file.createDataSet("/data/next secondary attack", nextSecondaryAttack, hdf5FlatCreateProps);
    file.createDataSet("/data/game time", gameTime, hdf5FlatCreateProps);
    file.createDataSet("/data/team", team, hdf5FlatCreateProps);
    file.createDataSet("/data/health", health, hdf5FlatCreateProps);
    file.createDataSet("/data/armor", armor, hdf5FlatCreateProps);
    file.createDataSet("/data/has helmet", hasHelmet, hdf5FlatCreateProps);
    file.createDataSet("/data/is alive", isAlive, hdf5FlatCreateProps);
    file.createDataSet("/data/ducking key pressed", duckingKeyPressed, hdf5FlatCreateProps);
    file.createDataSet("/data/duck amount", duckAmount, hdf5FlatCreateProps);
    file.createDataSet("/data/is reloading", isReloading, hdf5FlatCreateProps);
    file.createDataSet("/data/is walking", isWalking, hdf5FlatCreateProps);
    file.createDataSet("/data/is scoped", isScoped, hdf5FlatCreateProps);
    file.createDataSet("/data/is airborne", isAirborne, hdf5FlatCreateProps);
    file.createDataSet("/data/flash duration", flashDuration, hdf5FlatCreateProps);
    file.createDataSet("/data/active weapon", activeWeapon, hdf5FlatCreateProps);
    file.createDataSet("/data/primary weapon", primaryWeapon, hdf5FlatCreateProps);
    file.createDataSet("/data/primary bullets clip", primaryBulletsClip, hdf5FlatCreateProps);
    file.createDataSet("/data/primary bullets reserve", primaryBulletsReserve, hdf5FlatCreateProps);
    file.createDataSet("/data/secondary weapon", primaryWeapon, hdf5FlatCreateProps);
    file.createDataSet("/data/secondary bullets clip", primaryBulletsClip, hdf5FlatCreateProps);
    file.createDataSet("/data/secondary bullets reserve", primaryBulletsReserve, hdf5FlatCreateProps);
    file.createDataSet("/data/num HE", numHe, hdf5FlatCreateProps);
    file.createDataSet("/data/num flash", numFlash, hdf5FlatCreateProps);
    file.createDataSet("/data/num molotov", numMolotov, hdf5FlatCreateProps);
    file.createDataSet("/data/num incendiary", numIncendiary, hdf5FlatCreateProps);
    file.createDataSet("/data/num decoy", numDecoy, hdf5FlatCreateProps);
    file.createDataSet("/data/num zeus", numZeus, hdf5FlatCreateProps);
    file.createDataSet("/data/has defuser", hasDefuser, hdf5FlatCreateProps);
    file.createDataSet("/data/has bomb", hasBomb, hdf5FlatCreateProps);
    file.createDataSet("/data/money", money, hdf5FlatCreateProps);
    file.createDataSet("/data/ping", ping, hdf5FlatCreateProps);
}

void PlayerAtTick::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    playerId = file.getDataSet("/data/player id").read<std::vector<int64_t>>();
    tickId = file.getDataSet("/data/tick id").read<std::vector<int64_t>>();
    posX = file.getDataSet("/data/pos x").read<std::vector<double>>();
    posY = file.getDataSet("/data/pos y").read<std::vector<double>>();
    posZ = file.getDataSet("/data/pos z").read<std::vector<double>>();
    eyePosZ = file.getDataSet("/data/eye pos z").read<std::vector<double>>();
    velX = file.getDataSet("/data/vel x").read<std::vector<double>>();
    velY = file.getDataSet("/data/vel y").read<std::vector<double>>();
    velZ = file.getDataSet("/data/vel z").read<std::vector<double>>();
    viewX = file.getDataSet("/data/view x").read<std::vector<double>>();
    viewY = file.getDataSet("/data/view y").read<std::vector<double>>();
    aimPunchX = file.getDataSet("/data/aim punch x").read<std::vector<double>>();
    aimPunchY = file.getDataSet("/data/aim punch y").read<std::vector<double>>();
    viewPunchX = file.getDataSet("/data/view punch x").read<std::vector<double>>();
    viewPunchY = file.getDataSet("/data/view punch y").read<std::vector<double>>();
    recoilIndex = file.getDataSet("/data/recoil index").read<std::vector<double>>();
    nextPrimaryAttack = file.getDataSet("/data/next primary attack").read<std::vector<double>>();
    nextSecondaryAttack = file.getDataSet("/data/next secondary attack").read<std::vector<double>>();
    gameTime = file.getDataSet("/data/game time").read<std::vector<double>>();
    team = file.getDataSet("/data/team").read<std::vector<int16_t>>();
    health = file.getDataSet("/data/health").read<std::vector<double>>();
    armor = file.getDataSet("/data/armor").read<std::vector<double>>();
    hasHelmet = file.getDataSet("/data/has helmet").read<std::vector<bool>>();
    isAlive = file.getDataSet("/data/is alive").read<std::vector<bool>>();
    duckingKeyPressed = file.getDataSet("/data/ducking key pressed").read<std::vector<bool>>();
    duckAmount = file.getDataSet("/data/duck amount").read<std::vector<double>>();
    isReloading = file.getDataSet("/data/is reloading").read<std::vector<bool>>();
    isWalking = file.getDataSet("/data/is walking").read<std::vector<bool>>();
    isScoped = file.getDataSet("/data/is scoped").read<std::vector<bool>>();
    isAirborne = file.getDataSet("/data/is airborne").read<std::vector<bool>>();
    flashDuration = file.getDataSet("/data/flash duration").read<std::vector<double>>();
    activeWeapon = file.getDataSet("/data/active weapon").read<std::vector<int16_t>>();
    primaryWeapon = file.getDataSet("/data/primary weapon").read<std::vector<int16_t>>();
    primaryBulletsClip = file.getDataSet("/data/primary bullets clip").read<std::vector<int16_t>>();
    primaryBulletsReserve = file.getDataSet("/data/primary bullets reserve").read<std::vector<int16_t>>();
    secondaryWeapon = file.getDataSet("/data/secondary weapon").read<std::vector<int16_t>>();
    secondaryBulletsClip = file.getDataSet("/data/secondary bullets clip").read<std::vector<int16_t>>();
    secondaryBulletsReserve = file.getDataSet("/data/secondary bullets reserve").read<std::vector<int16_t>>();
    numHe = file.getDataSet("/data/num HE").read<std::vector<int16_t>>();
    numFlash = file.getDataSet("/data/num flash").read<std::vector<int16_t>>();
    numMolotov = file.getDataSet("/data/num molotov").read<std::vector<int16_t>>();
    numIncendiary = file.getDataSet("/data/num incendiary").read<std::vector<int16_t>>();
    numDecoy = file.getDataSet("/data/num decoy").read<std::vector<int16_t>>();
    numZeus = file.getDataSet("/data/num zeus").read<std::vector<int16_t>>();
    hasDefuser = file.getDataSet("/data/has defuser").read<std::vector<bool>>();
    hasBomb = file.getDataSet("/data/has bomb").read<std::vector<bool>>();
    money = file.getDataSet("/data/money").read<std::vector<int32_t>>();
    ping = file.getDataSet("/data/ping").read<std::vector<int32_t>>();
    size = static_cast<int64_t>(id.size());
}

void Spotted::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet("/data/spotted player", spottedPlayer, hdf5FlatCreateProps);
    file.createDataSet("/data/spotter player", spotterPlayer, hdf5FlatCreateProps);
    file.createDataSet("/data/is spotted", isSpotted, hdf5FlatCreateProps);
}

void Spotted::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    tickId = file.getDataSet("/data/tick id").read<std::vector<int64_t>>();
    spottedPlayer = file.getDataSet("/data/spotted player").read<std::vector<int64_t>>();
    spotterPlayer = file.getDataSet("/data/spotter player").read<std::vector<int64_t>>();
    isSpotted = file.getDataSet("/data/is spotted").read<std::vector<bool>>();
    size = static_cast<int64_t>(id.size());
}

void Footstep::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet("/data/stepping player", steppingPlayer, hdf5FlatCreateProps);
}

void Footstep::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    tickId = file.getDataSet("/data/tick id").read<std::vector<int64_t>>();
    steppingPlayer = file.getDataSet("/data/stepping player").read<std::vector<int64_t>>();
    size = static_cast<int64_t>(id.size());
}

void WeaponFire::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet("/data/shooter", shooter, hdf5FlatCreateProps);
    file.createDataSet("/data/weapon", shooter, hdf5FlatCreateProps);
}

void WeaponFire::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    tickId = file.getDataSet("/data/tick id").read<std::vector<int64_t>>();
    shooter = file.getDataSet("/data/shooter").read<std::vector<int64_t>>();
    weapon = file.getDataSet("/data/weapon").read<std::vector<int16_t>>();
    size = static_cast<int64_t>(id.size());
}

void Kills::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/tick id", tickId, hdf5FlatCreateProps);
    file.createDataSet("/data/killer", killer, hdf5FlatCreateProps);
    file.createDataSet("/data/victim", victim, hdf5FlatCreateProps);
    file.createDataSet("/data/weapon", weapon, hdf5FlatCreateProps);
    file.createDataSet("/data/assister", assister, hdf5FlatCreateProps);
    file.createDataSet("/data/isHeadshot", isHeadshot, hdf5FlatCreateProps);
    file.createDataSet("/data/isWallbang", isWallbang, hdf5FlatCreateProps);
    file.createDataSet("/data/penetrated objects", penetratedObjects, hdf5FlatCreateProps);
}

void Kills::fromHDF5Inner(HighFive::File & file) {
    id = file.getDataSet("/data/id").read<std::vector<int64_t>>();
    tickId = file.getDataSet("/data/tick id").read<std::vector<int64_t>>();
    killer = file.getDataSet("/data/killer").read<std::vector<int64_t>>();
    weapon = file.getDataSet("/data/weapon").read<std::vector<int64_t>>();
    weapon = file.getDataSet("/data/weapon").read<std::vector<int16_t>>();
    size = static_cast<int64_t>(id.size());
}
