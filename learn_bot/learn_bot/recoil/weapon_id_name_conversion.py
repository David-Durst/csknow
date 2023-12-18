from typing import Dict

# WARNING: THESE ARE DEPRECATED, WEAPON IDs FROM PARSER, NOT ENGINE, look at latent.place_area.simulation.constants
# for the better values from the engine and the weapon velocities

weapon_name_to_id: Dict[str, int] = {
    "EqUnknown": 0,

    # Pistols
    "P2000": 1,
    "Glock": 2,
    "P250": 3,
    "Deagle": 4,
    "FiveSeven": 5,
    "DualBerettas": 6,
    "Tec9": 7,
    "CZ": 8,
    "USP": 9,
    "Revolver": 10,
    "EQ_PISTOL_END": 11,

    # SMGs
    "MP7": 101,
    "MP9": 102,
    "Bizon": 103,
    "Mac10": 104,
    "UMP": 105,
    "P90": 106,
    "MP5": 107,
    "EQ_SMG_END": 108,

    # Heavy
    "SawedOff": 201,
    "Nova": 202,
    "Mag7": 203,
    "Swag7": 203,
    "XM1014": 204,
    "M249": 205,
    "Negev": 206,
    "EQ_HEAVY_END": 207,

    # Rifles
    "Galil": 301,
    "Famas": 302,
    "AK47": 303,
    "M4A4": 304,
    "M4A1": 305,
    "Scout": 306,
    "SSG08": 306,
    "SG556": 307,
    "SG553": 307,
    "AUG": 308,
    "AWP": 309,
    "Scar20": 310,
    "G3SG1": 311,
    "EQ_RIFLE_END": 312,

    # Equipment
    "EqZeus": 401,
    "EqKevlar": 402,
    "EqHelmet": 403,
    "EqBomb": 404,
    "EqKnife": 405,
    "EqDefuseKit": 406,
    "EqWorld": 407,

    # Grenades
    "EqDecoy": 501,
    "EqMolotov": 502,
    "EqIncendiary": 503,
    "EqFlash": 504,
    "EqSmoke": 505,
    "EqHE": 506,
}

weapon_id_to_name: Dict[int, str] = {}
for name, index in weapon_name_to_id.items():
    weapon_id_to_name[index] = name

