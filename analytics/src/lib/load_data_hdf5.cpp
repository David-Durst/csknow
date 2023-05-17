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
    auto idDataset = file.getDataSet("/data/id");
    id = idDataset.read<std::vector<int64_t>>();

    auto nameDataset = file.getDataSet("/data/name");
    name = nameDataset.read<std::vector<string>>();
}

void GameTypes::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/table type", tableType, hdf5FlatCreateProps);
}

void GameTypes::fromHDF5Inner(HighFive::File & file) {
    auto idDataset = file.getDataSet("/data/id");
    id = idDataset.read<std::vector<int64_t>>();

    auto tableTypeDataset = file.getDataSet("/data/table type");
    tableType = tableTypeDataset.read<std::vector<string>>();
}

void HitGroups::toHDF5Inner(HighFive::File & file) {
    HighFive::DataSetCreateProps hdf5FlatCreateProps;
    hdf5FlatCreateProps.add(HighFive::Deflate(6));
    hdf5FlatCreateProps.add(HighFive::Chunking(id.size()));

    file.createDataSet("/data/group name", groupName, hdf5FlatCreateProps);
}

void HitGroups::fromHDF5Inner(HighFive::File & file) {
    auto idDataset = file.getDataSet("/data/id");
    id = idDataset.read<std::vector<int64_t>>();

    auto groupNameDataset = file.getDataSet("/data/group name");
    groupName = groupNameDataset.read<std::vector<string>>();
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
    auto idDataset = file.getDataSet("/data/id");
    id = idDataset.read<std::vector<int64_t>>();

    auto demoFileDataset = file.getDataSet("/data/demo file");
    demoFile = demoFileDataset.read<std::vector<string>>();

    auto demoTickRateDataset = file.getDataSet("/data/demo tick rate");
    demoTickRate = demoTickRateDataset.read<std::vector<double>>();

    auto gameTickRateDataset = file.getDataSet("/data/game tick rate");
    gameTickRate = gameTickRateDataset.read<std::vector<double>>();

    auto mapNameDataset = file.getDataSet("/data/map name");
    mapName = mapNameDataset.read<std::vector<string>>();

    auto gameTypeDataset = file.getDataSet("/data/game type");
    gameType = gameTypeDataset.read<std::vector<int64_t>>();
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
    auto idDataset = file.getDataSet("/data/id");
    id = idDataset.read<std::vector<int64_t>>();

    auto gameIdDataset = file.getDataSet("/data/game id");
    gameId = gameIdDataset.read<std::vector<int64_t>>();

    auto nameDataset = file.getDataSet("/data/name");
    name = nameDataset.read<std::vector<string>>();

    auto steamIdDataset = file.getDataSet("/data/steam id");
    steamId = steamIdDataset.read<std::vector<int64_t>>();
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
    auto idDataset = file.getDataSet("/data/id");
    id = idDataset.read<std::vector<int64_t>>();

    auto gameIdDataset = file.getDataSet("/data/game id");
    gameId = gameIdDataset.read<std::vector<int64_t>>();

    auto startTickDataset = file.getDataSet("/data/start tick");
    startTick = startTickDataset.read<std::vector<int64_t>>();

    auto endTickDataset = file.getDataSet("/data/end tick");
    endTick = endTickDataset.read<std::vector<int64_t>>();

    auto endOfficialTickDataset = file.getDataSet("/data/end official tick");
    endOfficialTick = endOfficialTickDataset.read<std::vector<int64_t>>();

    auto warmupDataset = file.getDataSet("/data/warmup");
    warmup = warmupDataset.read<std::vector<bool>>();

    auto overtimeDataset = file.getDataSet("/data/overtime");
    overtime = overtimeDataset.read<std::vector<bool>>();

    auto freezeTimeEndDataset = file.getDataSet("/data/freeze time end");
    freezeTimeEnd = freezeTimeEndDataset.read<std::vector<int64_t>>();

    auto roundNumberDataset = file.getDataSet("/data/round number");
    roundNumber = roundNumberDataset.read<std::vector<int16_t>>();

    auto roundEndReasonDataset = file.getDataSet("/data/round end reason");
    roundEndReason = roundEndReasonDataset.read<std::vector<int16_t>>();

    auto winnerDataset = file.getDataSet("/data/winner");
    winner = winnerDataset.read<std::vector<int16_t>>();

    auto tWinsDataset = file.getDataSet("/data/t wins");
    tWins = tWinsDataset.read<std::vector<int16_t>>();

    auto ctWinsDataset = file.getDataSet("/data/ct wins");
    ctWins = ctWinsDataset.read<std::vector<int16_t>>();
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
    auto idDataset = file.getDataSet("/data/id");
    id = idDataset.read<std::vector<int64_t>>();

    auto roundIdDataset = file.getDataSet("/data/round id");
    roundId = roundIdDataset.read<std::vector<int64_t>>();

    auto gameTimeDataset = file.getDataSet("/data/game time");
    gameTime = gameTimeDataset.read<std::vector<int64_t>>();

    auto demoTickNumberDataset = file.getDataSet("/data/demo tick number");
    demoTickNumber = demoTickNumberDataset.read<std::vector<int64_t>>();

    auto gameTickNumberDataset = file.getDataSet("/data/game tick number");
    gameTickNumber = gameTickNumberDataset.read<std::vector<int64_t>>();

    auto bombCarrierDataset = file.getDataSet("/data/bomb carrier");
    bombCarrier = bombCarrierDataset.read<std::vector<int64_t>>();

    auto bombXDataset = file.getDataSet("/data/bomb x");
    bombX = bombXDataset.read<std::vector<double>>();

    auto bombYDataset = file.getDataSet("/data/bomb y");
    bombY = bombYDataset.read<std::vector<double>>();

    auto bombZDataset = file.getDataSet("/data/bomb z");
    bombZ = bombZDataset.read<std::vector<double>>();
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
    auto idDataset = file.getDataSet("/data/id");
    id = idDataset.read<std::vector<int64_t>>();

    auto playerIdDataset = file.getDataSet("/data/player id");
    playerId = playerIdDataset.read<std::vector<int64_t>>();

    auto tickIdDataset = file.getDataSet("/data/tick id");
    tickId = tickIdDataset.read<std::vector<int64_t>>();

    auto posXDataset = file.getDataSet("/data/pos x");
    posX = posXDataset.read<std::vector<double>>();

    auto posYDataset = file.getDataSet("/data/pos y");
    posY = posYDataset.read<std::vector<double>>();

    auto posZDataset = file.getDataSet("/data/pos z");
    posZ = posZDataset.read<std::vector<double>>();

    auto eyePosZDataset = file.getDataSet("/data/eye pos z");
    eyePosZ = eyePosZDataset.read<std::vector<double>>();

    auto velXDataset = file.getDataSet("/data/vel x");
    velX = velXDataset.read<std::vector<double>>();

    auto velYDataset = file.getDataSet("/data/vel y");
    velY = velYDataset.read<std::vector<double>>();

    auto velZDataset = file.getDataSet("/data/vel z");
    velZ = velZDataset.read<std::vector<double>>();

    auto viewXDataset = file.getDataSet("/data/view x");
    viewX = viewXDataset.read<std::vector<double>>();

    auto viewYDataset = file.getDataSet("/data/view y");
    viewY = viewYDataset.read<std::vector<double>>();

    auto aimPunchXDataset = file.getDataSet("/data/aim punch x");
    aimPunchX = aimPunchXDataset.read<std::vector<double>>();

    auto aimPunchYDataset = file.getDataSet("/data/aim punch y");
    aimPunchY = aimPunchYDataset.read<std::vector<double>>();

    auto viewPunchXDataset = file.getDataSet("/data/view punch x");
    viewPunchX = viewPunchXDataset.read<std::vector<double>>();

    auto viewPunchYDataset = file.getDataSet("/data/view punch y");
    viewPunchY = viewPunchYDataset.read<std::vector<double>>();

    auto recoilIndexDataset = file.getDataSet("/data/recoil index");
    recoilIndex = recoilIndexDataset.read<std::vector<double>>();

    auto nextPrimaryAttackDataset = file.getDataSet("/data/next primary attack");
    nextPrimaryAttack = nextPrimaryAttackDataset.read<std::vector<double>>();

    auto nextSecondaryAttackDataset = file.getDataSet("/data/next secondary attack");
    nextSecondaryAttack = nextSecondaryAttackDataset.read<std::vector<double>>();

    auto gameTimeDataset = file.getDataSet("/data/game time");
    gameTime = gameTimeDataset.read<std::vector<double>>();

    auto teamDataset = file.getDataSet("/data/team");
    team = teamDataset.read<std::vector<int16_t>>();

    auto healthDataset = file.getDataSet("/data/health");
    health = healthDataset.read<std::vector<double>>();

    auto armorDataset = file.getDataSet("/data/armor");
    armor = armorDataset.read<std::vector<double>>();

    auto hasHelmetDataset = file.getDataSet("/data/has helmet");
    hasHelmet = hasHelmetDataset.read<std::vector<bool>>();

    auto isAliveDataset = file.getDataSet("/data/is alive");
    isAlive = isAliveDataset.read<std::vector<bool>>();

    auto duckingKeyPressedDataset = file.getDataSet("/data/ducking key pressed");
    duckingKeyPressed = duckingKeyPressedDataset.read<std::vector<bool>>();

    auto duckAmountDataset = file.getDataSet("/data/ducking key pressed");
    duckingKeyPressed = duckingKeyPressedDataset.read<std::vector<bool>>();
}
