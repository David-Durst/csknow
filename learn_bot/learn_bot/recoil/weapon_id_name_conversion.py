from typing import Dict

weapon_name_to_id: Dict[str, int] = {
    "EqUnknown": 0,

    # Pistols
    "EqP2000": 1,
    "EqGlock": 2,
    "EqP250": 3,
    "EqDeagle": 4,
    "EqFiveSeven": 5,
    "EqDualBerettas": 6,
    "EqTec9": 7,
    "EqCZ": 8,
    "EqUSP": 9,
    "EqRevolver": 10,
    "EQ_PISTOL_END": 11,

    # SMGs
    "EqMP7": 101,
    "EqMP9": 102,
    "EqBizon": 103,
    "EqMac10": 104,
    "EqUMP": 105,
    "EqP90": 106,
    "EqMP5": 107,
    "EQ_SMG_END": 108,

    # Heavy
    "EqSawedOff": 201,
    "EqNova": 202,
    "EqMag7": 203,
    "EqSwag7": 203,
    "EqXM1014": 204,
    "EqM249": 205,
    "EqNegev": 206,
    "EQ_HEAVY_END": 207,

    # Rifles
    "EqGalil": 301,
    "EqFamas": 302,
    "EqAK47": 303,
    "EqM4A4": 304,
    "EqM4A1": 305,
    "EqScout": 306,
    "EqSSG08": 306,
    "EqSG556": 307,
    "EqSG553": 307,
    "EqAUG": 308,
    "EqAWP": 309,
    "EqScar20": 310,
    "EqG3SG1": 311,
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

