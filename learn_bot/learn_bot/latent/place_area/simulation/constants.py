from enum import Enum, unique, auto, IntEnum

import torch

from learn_bot.latent.place_area.pos_abs_from_delta_grid_or_radial import data_ticks_per_second, data_ticks_per_sim_tick

num_seconds_per_loop = 5
num_time_steps = data_ticks_per_second // data_ticks_per_sim_tick * num_seconds_per_loop


@unique
class EngineWeaponId(IntEnum):
    NoWeapon = 0
    Deagle = auto()
    Dualies = auto()
    FiveSeven = auto()
    Glock = auto()
    AK = 7
    AUG = auto()
    AWP = auto()
    FAMAS = auto()
    G3 = auto()
    Galil = 13
    M249 = auto()
    M4A4 = 16
    Mac10 = auto()
    P90 = 19
    MP5 = 23
    UMP = auto()
    XM1014 = auto()
    Bizon = auto()
    MAG7 = auto()
    Negev = auto()
    SawedOff = auto()
    Tec9 = auto()
    Zeus = auto()
    P2000 = auto()
    MP7 = auto()
    MP9 = auto()
    Nova = auto()
    P250 = auto()
    Scar = 38
    SG553 = auto()
    SSG = auto()
    Flashbang = 43
    HEGrenade = auto()
    Smoke = auto()
    Molotov = auto()
    Decoy = auto()
    Incendiary = auto()
    C4 = auto()
    M4A1S = 60
    USPS = auto()
    CZ = 63
    R8 = auto()

    def max_speed(self, scoped: bool) -> float:
        if self == EngineWeaponId.NoWeapon:
            return 250.
        if self == EngineWeaponId.Deagle:
            return 230.
        if self == EngineWeaponId.Dualies:
            return 240.
        if self == EngineWeaponId.FiveSeven:
            return 240.
        if self == EngineWeaponId.Glock:
            return 240.
        if self == EngineWeaponId.AK:
            return 215.
        if self == EngineWeaponId.AUG:
            if scoped:
                return 150.
            else:
                return 220.
        if self == EngineWeaponId.AWP:
            if scoped:
                return 100.
            else:
                return 200.
        if self == EngineWeaponId.FAMAS:
            return 220.
        if self == EngineWeaponId.G3:
            if scoped:
                return 120.
            else:
                return 215.
        if self == EngineWeaponId.Galil:
            return 215.
        if self == EngineWeaponId.M249:
            return 195.
        if self == EngineWeaponId.M4A4:
            return 225.
        if self == EngineWeaponId.Mac10:
            return 240.
        if self == EngineWeaponId.P90:
            return 230.
        if self == EngineWeaponId.MP5:
            return 235.
        if self == EngineWeaponId.UMP:
            return 230.
        if self == EngineWeaponId.XM1014:
            return 215.
        if self == EngineWeaponId.Bizon:
            return 240.
        if self == EngineWeaponId.MAG7:
            return 225.
        if self == EngineWeaponId.Negev:
            return 150.
        if self == EngineWeaponId.SawedOff:
            return 210.
        if self == EngineWeaponId.Tec9:
            return 240.
        if self == EngineWeaponId.Zeus:
            return 220.
        if self == EngineWeaponId.P2000:
            return 240.
        if self == EngineWeaponId.MP7:
            return 220.
        if self == EngineWeaponId.MP9:
            return 240.
        if self == EngineWeaponId.Nova:
            return 220.
        if self == EngineWeaponId.P250:
            return 240.
        if self == EngineWeaponId.Scar:
            if scoped:
                return 120.
            else:
                return 215.
        if self == EngineWeaponId.SG553:
            if scoped:
                return 150.
            else:
                return 210.
        if self == EngineWeaponId.SSG:
            return 230.
        if self == EngineWeaponId.Flashbang:
            return 245.
        if self == EngineWeaponId.HEGrenade:
            return 245.
        if self == EngineWeaponId.Smoke:
            return 245.
        if self == EngineWeaponId.Molotov:
            return 245.
        if self == EngineWeaponId.Decoy:
            return 245.
        if self == EngineWeaponId.Incendiary:
            return 245.
        if self == EngineWeaponId.C4:
            return 250.
        if self == EngineWeaponId.M4A1S:
            return 225.
        if self == EngineWeaponId.USPS:
            return 240.
        if self == EngineWeaponId.CZ:
            return 240.
        if self == EngineWeaponId.R8:
            return 220.

    def __str__(self) -> str:
        if self == EngineWeaponId.NoWeapon:
            return "None"
        if self == EngineWeaponId.Deagle:
            return "Deagle"
        if self == EngineWeaponId.Dualies:
            return "Dualies"
        if self == EngineWeaponId.FiveSeven:
            return "FiveSeven"
        if self == EngineWeaponId.Glock:
            return "Glock"
        if self == EngineWeaponId.AK:
            return "AK"
        if self == EngineWeaponId.AUG:
            return "AUG"
        if self == EngineWeaponId.AWP:
            return "AWP"
        if self == EngineWeaponId.FAMAS:
            return "FAMAS"
        if self == EngineWeaponId.G3:
            return "G3"
        if self == EngineWeaponId.Galil:
            return "Galil"
        if self == EngineWeaponId.M249:
            return "M249"
        if self == EngineWeaponId.M4A4:
            return "M4A4"
        if self == EngineWeaponId.Mac10:
            return "Mac10"
        if self == EngineWeaponId.P90:
            return "P90"
        if self == EngineWeaponId.MP5:
            return "MP5"
        if self == EngineWeaponId.UMP:
            return "UMP"
        if self == EngineWeaponId.XM1014:
            return "XM1014"
        if self == EngineWeaponId.Bizon:
            return "Bizon"
        if self == EngineWeaponId.MAG7:
            return "MAG7"
        if self == EngineWeaponId.Negev:
            return "Negev"
        if self == EngineWeaponId.SawedOff:
            return "SawedOff"
        if self == EngineWeaponId.Tec9:
            return "Tec9"
        if self == EngineWeaponId.Zeus:
            return "Zeus"
        if self == EngineWeaponId.P2000:
            return "P2000"
        if self == EngineWeaponId.MP7:
            return "MP7"
        if self == EngineWeaponId.MP9:
            return "MP9"
        if self == EngineWeaponId.Nova:
            return "Nova"
        if self == EngineWeaponId.P250:
            return "P250"
        if self == EngineWeaponId.Scar:
            return "Scar"
        if self == EngineWeaponId.SG553:
            return "SG553"
        if self == EngineWeaponId.SSG:
            return "SSG"
        if self == EngineWeaponId.Flashbang:
            return "Flashbang"
        if self == EngineWeaponId.HEGrenade:
            return "HEGrenade"
        if self == EngineWeaponId.Smoke:
            return "Smoke"
        if self == EngineWeaponId.Molotov:
            return "Molotov"
        if self == EngineWeaponId.Decoy:
            return "Decoy"
        if self == EngineWeaponId.Incendiary:
            return "Incendiary"
        if self == EngineWeaponId.C4:
            return "C4"
        if self == EngineWeaponId.M4A1S:
            return "M4A1S"
        if self == EngineWeaponId.USPS:
            return "USPS"
        if self == EngineWeaponId.CZ:
            return "CZ"
        if self == EngineWeaponId.R8:
            return "R8"


# since pytorch doesn't support nested index_select, flatten it and I'll do lookup index computation my self
weapon_scoped_to_max_speed_tensor: torch.Tensor = torch.zeros([2 * (int(EngineWeaponId.R8) + 1)])
for engine_weapon_id in EngineWeaponId:
    weapon_scoped_to_max_speed_tensor[2 * int(engine_weapon_id)] = engine_weapon_id.max_speed(False)
    weapon_scoped_to_max_speed_tensor[2 * int(engine_weapon_id) + 1] = engine_weapon_id.max_speed(True)

