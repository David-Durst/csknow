export interface GrenadesRow {
    thrower: string;
    grenadeType: string;
    tickNumber: bigint;
    demoFile: string;
}

export interface KillsRow {
    killer: string;
    victim: string;
    weapon: string;
    assister: string;
    isHeadshot: boolean;
    isWallbang: boolean;
    penetratedObjects: string;
    tickNumber: bigint;
    demoFile: string;
}

export class GameData {
    grenades: GrenadesRow[];
    kills: KillsRow[];
}
