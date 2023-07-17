//
// Created by durst on 2/21/23.
//

#ifndef CSKNOW_WEAPON_ID_CONVERTER_H
#define CSKNOW_WEAPON_ID_CONVERTER_H
// https://tf2b.com/itemlist.php?gid=730

#include <cstdint>
#include "queries/parser_constants.h"
#include <string>

enum class EngineWeaponId {
    None = 0,
    Deagle,
    Dualies,
    FiveSeven,
    Glock,
    AK = 7,
    AUG,
    AWP,
    FAMAS,
    G3,
    Galil = 13,
    M249,
    M4A4 = 16,
    Mac10,
    P90 = 19,
    MP5 = 23,
    UMP,
    XM1014,
    Bizon,
    MAG7,
    Negev,
    SawedOff,
    Tec9,
    Zeus,
    P2000,
    MP7,
    MP9,
    Nova,
    P250,
    Scar = 38,
    SG553,
    SSG,
    Flashbang = 43,
    HEGrenade,
    Smoke,
    Molotov,
    Decoy,
    Incendiary,
    C4,
    M4A1S = 60,
    USPS,
    CZ = 63,
    R8
};

EngineWeaponId demoEquipmentTypeToEngineWeaponId(int16_t demoEquipmentType);
EngineWeaponId demoEquipmentTypeToEngineWeaponId(DemoEquipmentType demoEquipmentType);
std::string demoEquipmentTypeToString(int16_t demoEquipmentType);
std::string demoEquipmentTypeToString(DemoEquipmentType demoEquipmentType);

#endif //CSKNOW_WEAPON_ID_CONVERTER_H
