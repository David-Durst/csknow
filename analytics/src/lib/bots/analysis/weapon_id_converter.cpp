//
// Created by durst on 2/21/23.
//

#include <string>
#include "bots/analysis/weapon_id_converter.h"
#include "enum_helpers.h"

EngineWeaponId demoEquipmentTypeToEngineWeaponId(int16_t demoEquipmentType) {
    return demoEquipmentTypeToEngineWeaponId(static_cast<DemoEquipmentType>(demoEquipmentType));
}

EngineWeaponId demoEquipmentTypeToEngineWeaponId(DemoEquipmentType demoEquipmentType) {
    switch (demoEquipmentType) {
        case DemoEquipmentType::EqUnknown:
            return EngineWeaponId::None;
        case DemoEquipmentType::EqP2000:
            return EngineWeaponId::P2000;
        case DemoEquipmentType::EqGlock:
            return EngineWeaponId::Glock;
        case DemoEquipmentType::EqP250:
            return EngineWeaponId::P250;
        case DemoEquipmentType::EqDeagle:
            return EngineWeaponId::Deagle;
        case DemoEquipmentType::EqFiveSeven:
            return EngineWeaponId::FiveSeven;
        case DemoEquipmentType::EqDualBerettas:
            return EngineWeaponId::Dualies;
        case DemoEquipmentType::EqTec9:
            return EngineWeaponId::Tec9;
        case DemoEquipmentType::EqCZ:
            return EngineWeaponId::CZ;
        case DemoEquipmentType::EqUSP:
            return EngineWeaponId::USPS;
        case DemoEquipmentType::EqRevolver:
            return EngineWeaponId::R8;

        case DemoEquipmentType::EqMP7:
            return EngineWeaponId::MP7;
        case DemoEquipmentType::EqMP9:
            return EngineWeaponId::MP9;
        case DemoEquipmentType::EqBizon:
            return EngineWeaponId::Bizon;
        case DemoEquipmentType::EqMac10:
            return EngineWeaponId::Mac10;
        case DemoEquipmentType::EqUMP:
            return EngineWeaponId::UMP;
        case DemoEquipmentType::EqP90:
            return EngineWeaponId::P90;
        case DemoEquipmentType::EqMP5:
            return EngineWeaponId::MP5;

        case DemoEquipmentType::EqSawedOff:
            return EngineWeaponId::SawedOff;
        case DemoEquipmentType::EqNova:
            return EngineWeaponId::Nova;
        case DemoEquipmentType::EqMag7:
            return EngineWeaponId::MAG7;
        case DemoEquipmentType::EqXM1014:
            return EngineWeaponId::XM1014;
        case DemoEquipmentType::EqM249:
            return EngineWeaponId::M249;
        case DemoEquipmentType::EqNegev:
            return EngineWeaponId::Negev;

        case DemoEquipmentType::EqGalil:
            return EngineWeaponId::Galil;
        case DemoEquipmentType::EqFamas:
            return EngineWeaponId::FAMAS;
        case DemoEquipmentType::EqAK47:
            return EngineWeaponId::AK;
        case DemoEquipmentType::EqM4A4:
            return EngineWeaponId::M4A4;
        case DemoEquipmentType::EqM4A1:
            return EngineWeaponId::M4A1S;
        case DemoEquipmentType::EqScout:
            return EngineWeaponId::SSG;
        case DemoEquipmentType::EqSG553:
            return EngineWeaponId::SG553;
        case DemoEquipmentType::EqAUG:
            return EngineWeaponId::AUG;
        case DemoEquipmentType::EqAWP:
            return EngineWeaponId::AWP;
        case DemoEquipmentType::EqScar20:
            return EngineWeaponId::Scar;
        case DemoEquipmentType::EqG3SG1:
            return EngineWeaponId::G3;

        case DemoEquipmentType::EqBomb:
            return EngineWeaponId::C4;

        case DemoEquipmentType::EqDecoy:
            return EngineWeaponId::Decoy;
        case DemoEquipmentType::EqMolotov:
            return EngineWeaponId::Molotov;
        case DemoEquipmentType::EqIncendiary:
            return EngineWeaponId::Incendiary;
        case DemoEquipmentType::EqFlash:
            return EngineWeaponId::Flashbang;
        case DemoEquipmentType::EqSmoke:
            return EngineWeaponId::Smoke;
        case DemoEquipmentType::EqHE:
            return EngineWeaponId::HEGrenade;


        default:
            return EngineWeaponId::None;
    }
}

std::string demoEquipmentTypeToString(int16_t demoEquipmentType) {
    return demoEquipmentTypeToString(static_cast<DemoEquipmentType>(demoEquipmentType));
}

std::string demoEquipmentTypeToString(DemoEquipmentType demoEquipmentType) {
    switch (demoEquipmentType) {
        case DemoEquipmentType::EqUnknown:
            return "None";
        case DemoEquipmentType::EqP2000:
            return "P2000";
        case DemoEquipmentType::EqGlock:
            return "Glock";
        case DemoEquipmentType::EqP250:
            return "P250";
        case DemoEquipmentType::EqDeagle:
            return "Deagle";
        case DemoEquipmentType::EqFiveSeven:
            return "FiveSeven";
        case DemoEquipmentType::EqDualBerettas:
            return "Dualies";
        case DemoEquipmentType::EqTec9:
            return "Tec9";
        case DemoEquipmentType::EqCZ:
            return "CZ";
        case DemoEquipmentType::EqUSP:
            return "USPS";
        case DemoEquipmentType::EqRevolver:
            return "R8";

        case DemoEquipmentType::EqMP7:
            return "MP7";
        case DemoEquipmentType::EqMP9:
            return "MP9";
        case DemoEquipmentType::EqBizon:
            return "Bizon";
        case DemoEquipmentType::EqMac10:
            return "Mac10";
        case DemoEquipmentType::EqUMP:
            return "UMP";
        case DemoEquipmentType::EqP90:
            return "P90";
        case DemoEquipmentType::EqMP5:
            return "MP5";

        case DemoEquipmentType::EqSawedOff:
            return "SawedOff";
        case DemoEquipmentType::EqNova:
            return "Nova";
        case DemoEquipmentType::EqMag7:
            return "MAG7";
        case DemoEquipmentType::EqXM1014:
            return "XM1014";
        case DemoEquipmentType::EqM249:
            return "M249";
        case DemoEquipmentType::EqNegev:
            return "Negev";

        case DemoEquipmentType::EqGalil:
            return "Galil";
        case DemoEquipmentType::EqFamas:
            return "FAMAS";
        case DemoEquipmentType::EqAK47:
            return "AK";
        case DemoEquipmentType::EqM4A4:
            return "M4A4";
        case DemoEquipmentType::EqM4A1:
            return "M4A1S";
        case DemoEquipmentType::EqScout:
            return "SSG";
        case DemoEquipmentType::EqSG553:
            return "SG553";
        case DemoEquipmentType::EqAUG:
            return "AUG";
        case DemoEquipmentType::EqAWP:
            return "AWP";
        case DemoEquipmentType::EqScar20:
            return "Scar";
        case DemoEquipmentType::EqG3SG1:
            return "G3";

        case DemoEquipmentType::EqBomb:
            return "C4";

        case DemoEquipmentType::EqDecoy:
            return "Decoy";
        case DemoEquipmentType::EqMolotov:
            return "Molotov";
        case DemoEquipmentType::EqIncendiary:
            return "Incendiary";
        case DemoEquipmentType::EqFlash:
            return "Flashbang";
        case DemoEquipmentType::EqSmoke:
            return "Smoke";
        case DemoEquipmentType::EqHE:
            return "HEGrenade";


        default:
            return "None";
    }
}
