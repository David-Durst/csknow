const utf8Decoder = new TextDecoder("utf-8");

function parseBool(b: string) {
    return b == "1";
}

export class PositionRow {
    tickNumber: number;
    matchStarted: boolean;
    gamePhase: number;
    roundsPlayed: number;
    isWarmup: boolean;
    roundStart: boolean;
    roundEnd: boolean
    roundEndReason: number;
    freezeTimeEnded: boolean
    tScore: number;
    ctScore: number;
    numPlayers: number;
    player0Name: string;
    player0Team: number;
    player0XPosition: number;
    player0YPosition: number;
    player0ZPosition: number;
    player0XViewDirection: number;
    player0YViewDirection: number;
    player0IsAlive: boolean;
    // remove this later when fix data set
    player0TeamRepeat: number;
    player0isBlinded: boolean;
    player1Name: string;
    player1Team: number;
    player1XPosition: number;
    player1YPosition: number;
    player1ZPosition: number;
    player1XViewDirection: number;
    player1YViewDirection: number;
    player1IsAlive: boolean;
    // remove this later when fix data set
    player1TeamRepeat: number;
    player1isBlinded: boolean;
    player2Name: string;
    player2Team: number;
    player2XPosition: number;
    player2YPosition: number;
    player2ZPosition: number;
    player2XViewDirection: number;
    player2YViewDirection: number;
    player2IsAlive: boolean;
    // remove this later when fix data set
    player2TeamRepeat: number;
    player2isBlinded: boolean;
    player3Name: string;
    player3Team: number;
    player3XPosition: number;
    player3YPosition: number;
    player3ZPosition: number;
    player3XViewDirection: number;
    player3YViewDirection: number;
    player3IsAlive: boolean;
    // remove this later when fix data set
    player3TeamRepeat: number;
    player3isBlinded: boolean;
    player4Name: string;
    player4Team: number;
    player4XPosition: number;
    player4YPosition: number;
    player4ZPosition: number;
    player4XViewDirection: number;
    player4YViewDirection: number;
    player4IsAlive: boolean;
    // remove this later when fix data set
    player4TeamRepeat: number;
    player4isBlinded: boolean;
    player5Name: string;
    player5Team: number;
    player5XPosition: number;
    player5YPosition: number;
    player5ZPosition: number;
    player5XViewDirection: number;
    player5YViewDirection: number;
    player5IsAlive: boolean;
    // remove this later when fix data set
    player5TeamRepeat: number;
    player5isBlinded: boolean;
    player6Name: string;
    player6Team: number;
    player6XPosition: number;
    player6YPosition: number;
    player6ZPosition: number;
    player6XViewDirection: number;
    player6YViewDirection: number;
    player6IsAlive: boolean;
    // remove this later when fix data set
    player6TeamRepeat: number;
    player6isBlinded: boolean;
    player7Name: string;
    player7Team: number;
    player7XPosition: number;
    player7YPosition: number;
    player7ZPosition: number;
    player7XViewDirection: number;
    player7YViewDirection: number;
    player7IsAlive: boolean;
    // remove this later when fix data set
    player7TeamRepeat: number;
    player7isBlinded: boolean;
    player8Name: string;
    player8Team: number;
    player8XPosition: number;
    player8YPosition: number;
    player8ZPosition: number;
    player8XViewDirection: number;
    player8YViewDirection: number;
    player8IsAlive: boolean;
    // remove this later when fix data set
    player8TeamRepeat: number;
    player8isBlinded: boolean;
    player9Name: string;
    player9Team: number;
    player9XPosition: number;
    player9YPosition: number;
    player9ZPosition: number;
    player9XViewDirection: number;
    player9YViewDirection: number;
    player9IsAlive: boolean;
    // remove this later when fix data set
    player9TeamRepeat: number;
    player9isBlinded: boolean;
    demoFile: string;

    constructor(tickNumber: number, matchStarted: boolean, gamePhase: number,
                roundsPlayed: number, isWarmup: boolean, roundStart: boolean, roundEnd: boolean,
                roundEndReason: number, freezeTimeEnded: boolean, tScore: number, ctScore: number, numPlayers: number,
                player0Name: string, player0Team: number, player0XPosition: number, player0YPosition: number,
                player0ZPosition: number, player0XViewDirection: number, player0YViewDirection: number,
                player0IsAlive: boolean, player0TeamRepeat: number, player0isBlinded: boolean,
                player1Name: string, player1Team: number, player1XPosition: number, player1YPosition: number,
                player1ZPosition: number, player1XViewDirection: number, player1YViewDirection: number,
                player1IsAlive: boolean, player1TeamRepeat: number, player1isBlinded: boolean,
                player2Name: string, player2Team: number, player2XPosition: number, player2YPosition: number,
                player2ZPosition: number, player2XViewDirection: number, player2YViewDirection: number,
                player2IsAlive: boolean, player2TeamRepeat: number, player2isBlinded: boolean,
                player3Name: string, player3Team: number, player3XPosition: number, player3YPosition: number,
                player3ZPosition: number, player3XViewDirection: number, player3YViewDirection: number,
                player3IsAlive: boolean, player3TeamRepeat: number, player3isBlinded: boolean,
                player4Name: string, player4Team: number, player4XPosition: number, player4YPosition: number,
                player4ZPosition: number, player4XViewDirection: number, player4YViewDirection: number,
                player4IsAlive: boolean, player4TeamRepeat: number, player4isBlinded: boolean,
                player5Name: string, player5Team: number, player5XPosition: number, player5YPosition: number,
                player5ZPosition: number, player5XViewDirection: number, player5YViewDirection: number,
                player5IsAlive: boolean, player5TeamRepeat: number, player5isBlinded: boolean,
                player6Name: string, player6Team: number, player6XPosition: number, player6YPosition: number,
                player6ZPosition: number, player6XViewDirection: number, player6YViewDirection: number,
                player6IsAlive: boolean, player6TeamRepeat: number, player6isBlinded: boolean,
                player7Name: string, player7Team: number, player7XPosition: number, player7YPosition: number,
                player7ZPosition: number, player7XViewDirection: number, player7YViewDirection: number,
                player7IsAlive: boolean, player7TeamRepeat: number, player7isBlinded: boolean,
                player8Name: string, player8Team: number, player8XPosition: number, player8YPosition: number,
                player8ZPosition: number, player8XViewDirection: number, player8YViewDirection: number,
                player8IsAlive: boolean, player8TeamRepeat: number, player8isBlinded: boolean,
                player9Name: string, player9Team: number, player9XPosition: number, player9YPosition: number,
                player9ZPosition: number, player9XViewDirection: number, player9YViewDirection: number,
                player9IsAlive: boolean, player9TeamRepeat: number, player9isBlinded: boolean,
                demoFile: string
                ) {
        this.tickNumber = tickNumber;
        this.matchStarted = matchStarted;
        this.gamePhase = gamePhase;
        this.roundsPlayed = roundsPlayed;
        this.isWarmup = isWarmup;
        this.roundStart = roundStart;
        this.roundEnd = roundEnd;
        this.roundEndReason = roundEndReason;
        this.freezeTimeEnded = freezeTimeEnded;
        this.tScore = tScore;
        this.ctScore = ctScore;
        this.numPlayers = numPlayers;
        this.player0Name = player0Name;
        this.player0Team = player0Team;
        this.player0XPosition = player0XPosition;
        this.player0YPosition = player0YPosition;
        this.player0ZPosition = player0ZPosition;
        this.player0XViewDirection = player0XViewDirection;
        this.player0YViewDirection = player0YViewDirection;
        this.player0IsAlive = player0IsAlive;
        // remove this later when fix data set
        this.player0TeamRepeat = player0TeamRepeat;
        this.player0isBlinded = player0isBlinded;
        this.player1Name = player1Name;
        this.player1Team = player1Team;
        this.player1XPosition = player1XPosition;
        this.player1YPosition = player1YPosition;
        this.player1ZPosition = player1ZPosition;
        this.player1XViewDirection = player1XViewDirection;
        this.player1YViewDirection = player1YViewDirection;
        this.player1IsAlive = player1IsAlive;
        // remove this later when fix data set
        this.player1TeamRepeat = player1TeamRepeat;
        this.player1isBlinded = player1isBlinded;
        this.player2Name = player2Name;
        this.player2Team = player2Team;
        this.player2XPosition = player2XPosition;
        this.player2YPosition = player2YPosition;
        this.player2ZPosition = player2ZPosition;
        this.player2XViewDirection = player2XViewDirection;
        this.player2YViewDirection = player2YViewDirection;
        this.player2IsAlive = player2IsAlive;
        // remove this later when fix data set
        this.player2TeamRepeat = player2TeamRepeat;
        this.player2isBlinded = player2isBlinded;
        this.player3Name = player3Name;
        this.player3Team = player3Team;
        this.player3XPosition = player3XPosition;
        this.player3YPosition = player3YPosition;
        this.player3ZPosition = player3ZPosition;
        this.player3XViewDirection = player3XViewDirection;
        this.player3YViewDirection = player3YViewDirection;
        this.player3IsAlive = player3IsAlive;
        // remove this later when fix data set
        this.player3TeamRepeat = player3TeamRepeat;
        this.player3isBlinded = player3isBlinded;
        this.player4Name = player4Name;
        this.player4Team = player4Team;
        this.player4XPosition = player4XPosition;
        this.player4YPosition = player4YPosition;
        this.player4ZPosition = player4ZPosition;
        this.player4XViewDirection = player4XViewDirection;
        this.player4YViewDirection = player4YViewDirection;
        this.player4IsAlive = player4IsAlive;
        // remove this later when fix data set
        this.player4TeamRepeat = player4TeamRepeat;
        this.player4isBlinded = player4isBlinded;
        this.player5Name = player5Name;
        this.player5Team = player5Team;
        this.player5XPosition = player5XPosition;
        this.player5YPosition = player5YPosition;
        this.player5ZPosition = player5ZPosition;
        this.player5XViewDirection = player5XViewDirection;
        this.player5YViewDirection = player5YViewDirection;
        this.player5IsAlive = player5IsAlive;
        // remove this later when fix data set
        this.player5TeamRepeat = player5TeamRepeat;
        this.player5isBlinded = player5isBlinded;
        this.player6Name = player6Name;
        this.player6Team = player6Team;
        this.player6XPosition = player6XPosition;
        this.player6YPosition = player6YPosition;
        this.player6ZPosition = player6ZPosition;
        this.player6XViewDirection = player6XViewDirection;
        this.player6YViewDirection = player6YViewDirection;
        this.player6IsAlive = player6IsAlive;
        // remove this later when fix data set
        this.player6TeamRepeat = player6TeamRepeat;
        this.player6isBlinded = player6isBlinded;
        this.player7Name = player7Name;
        this.player7Team = player7Team;
        this.player7XPosition = player7XPosition;
        this.player7YPosition = player7YPosition;
        this.player7ZPosition = player7ZPosition;
        this.player7XViewDirection = player7XViewDirection;
        this.player7YViewDirection = player7YViewDirection;
        this.player7IsAlive = player7IsAlive;
        // remove this later when fix data set
        this.player7TeamRepeat = player7TeamRepeat;
        this.player7isBlinded = player7isBlinded;
        this.player8Name = player8Name;
        this.player8Team = player8Team;
        this.player8XPosition = player8XPosition;
        this.player8YPosition = player8YPosition;
        this.player8ZPosition = player8ZPosition;
        this.player8XViewDirection = player8XViewDirection;
        this.player8YViewDirection = player8YViewDirection;
        this.player8IsAlive = player8IsAlive;
        // remove this later when fix data set
        this.player8TeamRepeat = player8TeamRepeat;
        this.player8isBlinded = player8isBlinded;
        this.player9Name = player9Name;
        this.player9Team = player9Team;
        this.player9XPosition = player9XPosition;
        this.player9YPosition = player9YPosition;
        this.player9ZPosition = player9ZPosition;
        this.player9XViewDirection = player9XViewDirection;
        this.player9YViewDirection = player9YViewDirection;
        this.player9IsAlive = player9IsAlive;
        // remove this later when fix data set
        this.player9TeamRepeat = player9TeamRepeat;
        this.player9isBlinded = player9isBlinded;
        this.demoFile = demoFile;

    }

}

export async function parsePosition(tuple: { value: Uint8Array; done: boolean; }) {
    const linesUnsplit = tuple.value ? utf8Decoder.decode(tuple.value, {stream: true}) : "";
    const lines = linesUnsplit.split("\n");
    for (let lineNumber = 1; lineNumber < lines.length; lineNumber++) {
        if (lines[lineNumber].trim() === "") {
            continue;
        }
        let currentLine = lines[lineNumber].split(",");
        gameData.position.push(new PositionRow(
            // first 12 aren't palyer specified
            parseInt(currentLine[0]), parseBool(currentLine[1]), parseInt(currentLine[2]),
            parseInt(currentLine[3]), parseBool(currentLine[4]), parseBool(currentLine[5]),
            parseBool(currentLine[6]), parseInt(currentLine[7]), parseBool(currentLine[8]),
            parseInt(currentLine[9]), parseInt(currentLine[10]), parseInt(currentLine[11]),
            // each player is 10 entries
            // player 0
            currentLine[11], parseInt(currentLine[12]), parseFloat(currentLine[13]),
            parseFloat(currentLine[13]), parseFloat(currentLine[14]), parseFloat(currentLine[15]),
            parseFloat(currentLine[16]), parseBool(currentLine[18]), parseInt(currentLine[19]), parseBool(currentLine[20]),
            // player 1
            currentLine[21], parseInt(currentLine[22]), parseFloat(currentLine[23]),
            parseFloat(currentLine[23]), parseFloat(currentLine[24]), parseFloat(currentLine[25]),
            parseFloat(currentLine[26]), parseBool(currentLine[28]), parseInt(currentLine[29]), parseBool(currentLine[30]),
            // player 2
            currentLine[31], parseInt(currentLine[32]), parseFloat(currentLine[33]),
            parseFloat(currentLine[33]), parseFloat(currentLine[34]), parseFloat(currentLine[35]),
            parseFloat(currentLine[36]), parseBool(currentLine[38]), parseInt(currentLine[39]), parseBool(currentLine[40]),
            // player 3
            currentLine[41], parseInt(currentLine[42]), parseFloat(currentLine[43]),
            parseFloat(currentLine[43]), parseFloat(currentLine[44]), parseFloat(currentLine[45]),
            parseFloat(currentLine[46]), parseBool(currentLine[48]), parseInt(currentLine[49]), parseBool(currentLine[50]),
            // player 4
            currentLine[51], parseInt(currentLine[52]), parseFloat(currentLine[53]),
            parseFloat(currentLine[53]), parseFloat(currentLine[54]), parseFloat(currentLine[55]),
            parseFloat(currentLine[56]), parseBool(currentLine[58]), parseInt(currentLine[59]), parseBool(currentLine[60]),
            // player 5
            currentLine[61], parseInt(currentLine[62]), parseFloat(currentLine[63]),
            parseFloat(currentLine[63]), parseFloat(currentLine[64]), parseFloat(currentLine[65]),
            parseFloat(currentLine[66]), parseBool(currentLine[68]), parseInt(currentLine[69]), parseBool(currentLine[70]),
            // player 6
            currentLine[71], parseInt(currentLine[72]), parseFloat(currentLine[73]),
            parseFloat(currentLine[73]), parseFloat(currentLine[74]), parseFloat(currentLine[75]),
            parseFloat(currentLine[76]), parseBool(currentLine[78]), parseInt(currentLine[79]), parseBool(currentLine[80]),
            // player 7
            currentLine[81], parseInt(currentLine[82]), parseFloat(currentLine[83]),
            parseFloat(currentLine[83]), parseFloat(currentLine[84]), parseFloat(currentLine[85]),
            parseFloat(currentLine[86]), parseBool(currentLine[88]), parseInt(currentLine[89]), parseBool(currentLine[90]),
            // player 8
            currentLine[91], parseInt(currentLine[92]), parseFloat(currentLine[93]),
            parseFloat(currentLine[93]), parseFloat(currentLine[94]), parseFloat(currentLine[95]),
            parseFloat(currentLine[96]), parseBool(currentLine[98]), parseInt(currentLine[99]), parseBool(currentLine[100]),
            // player 9
            currentLine[101], parseInt(currentLine[102]), parseFloat(currentLine[103]),
            parseFloat(currentLine[103]), parseFloat(currentLine[104]), parseFloat(currentLine[105]),
            parseFloat(currentLine[106]), parseBool(currentLine[108]), parseInt(currentLine[109]), parseBool(currentLine[110]),
            // after player data
            currentLine[111]
        ));
        //console.log("length of position:" + gameData.position.length.toString() + ", line number: " + lineNumber.toString())
    }
    if (!tuple.done) {
        await positionReader.read().then(parsePosition);
    }
}

export let positionReader:any = null;
export function setPositionReader(readerInput: any) {
    positionReader = readerInput
}

export class SpottedRow {
    spottedPlayer: string;
    player0Name: string;
    player0Spotter: boolean;
    player1Name: string;
    player1Spotter: boolean;
    player2Name: string;
    player2Spotter: boolean;
    player3Name: string;
    player3Spotter: boolean;
    player4Name: string;
    player4Spotter: boolean;
    player5Name: string;
    player5Spotter: boolean;
    player6Name: string;
    player6Spotter: boolean;
    player7Name: string;
    player7Spotter: boolean;
    player8Name: string;
    player8Spotter: boolean;
    player9Name: string;
    player9Spotter: boolean;
    tickNumber: number;
    demoFile: string;
    
    constructor(spottedPlayer: string, player0Name: string, player0Spotter: boolean, 
                player1Name: string, player1Spotter: boolean, 
                player2Name: string, player2Spotter: boolean, 
                player3Name: string, player3Spotter: boolean, 
                player4Name: string, player4Spotter: boolean, 
                player5Name: string, player5Spotter: boolean, 
                player6Name: string, player6Spotter: boolean, 
                player7Name: string, player7Spotter: boolean, 
                player8Name: string, player8Spotter: boolean, 
                player9Name: string, player9Spotter: boolean, 
                tickNumber: number, demoFile: string) {
        this.spottedPlayer = spottedPlayer;
        this.player0Name = player0Name;
        this.player0Spotter = player0Spotter;
        this.player1Name = player1Name;
        this.player1Spotter = player1Spotter;
        this.player2Name = player2Name;
        this.player2Spotter = player2Spotter;
        this.player3Name = player3Name;
        this.player3Spotter = player3Spotter;
        this.player4Name = player4Name;
        this.player4Spotter = player4Spotter;
        this.player5Name = player5Name;
        this.player5Spotter = player5Spotter;
        this.player6Name = player6Name;
        this.player6Spotter = player6Spotter;
        this.player7Name = player7Name;
        this.player7Spotter = player7Spotter;
        this.player8Name = player8Name;
        this.player8Spotter = player8Spotter;
        this.player9Name = player9Name;
        this.player9Spotter = player9Spotter;
        this.tickNumber = tickNumber;
        this.demoFile = demoFile;
    }
}

export async function parseSpotted(tuple: { value: Uint8Array; done: boolean; }) {
    const linesUnsplit = tuple.value ? utf8Decoder.decode(tuple.value, {stream: true}) : "";
    const lines = linesUnsplit.split("\n");
    for(let lineNumber = 1; lineNumber < lines.length; lineNumber++) {
        if (lines[lineNumber].trim() === "") {
            continue;
        }
        let currentLine = lines[lineNumber].split(",");
        gameData.spotted.push(new SpottedRow(
            currentLine[0], currentLine[1], parseBool(currentLine[2]),
            currentLine[3], parseBool(currentLine[4]), currentLine[5], parseBool(currentLine[6]),
            currentLine[7], parseBool(currentLine[8]), currentLine[9], parseBool(currentLine[10]),
            currentLine[11], parseBool(currentLine[12]), currentLine[13], parseBool(currentLine[13]),
            currentLine[14], parseBool(currentLine[15]), currentLine[16], parseBool(currentLine[18]),
            currentLine[16], parseBool(currentLine[18]), parseInt(currentLine[2]), currentLine[3]
        ));
    }
    if (!tuple.done) {
        await spottedReader.read().then(parseSpotted);
    }
}

export let spottedReader:any = null;
export function setSpottedReader(readerInput: any) {
    spottedReader = readerInput
}

export class WeaponFireRow {
    shooter: string;
    weapon: string;
    tickNumber: number;
    demoFile: string;
    
    constructor(shooter: string, weapon: string, 
                tickNumber: number, demoFile: string) {
        this.shooter = shooter;
        this.weapon = weapon;
        this.tickNumber = tickNumber;
        this.demoFile = demoFile;
    }
}

export async function parseWeaponFire(tuple: { value: Uint8Array; done: boolean; }) {
    const linesUnsplit = tuple.value ? utf8Decoder.decode(tuple.value, {stream: true}) : "";
    const lines = linesUnsplit.split("\n");
    for(let lineNumber = 1; lineNumber < lines.length; lineNumber++) {
        if (lines[lineNumber].trim() === "") {
            continue;
        }
        let currentLine = lines[lineNumber].split(",");
        gameData.weaponFire.push(new WeaponFireRow(
            currentLine[0], currentLine[1], parseInt(currentLine[2]), currentLine[3]
        ));
    }
    if (!tuple.done) {
        await weaponFireReader.read().then(parseWeaponFire);
    }
}

export let weaponFireReader:any = null;
export function setWeaponFireReader(readerInput: any) {
    weaponFireReader = readerInput
}

export class PlayerHurtRow {
    victimName: string;
    armorDamage: number;
    armor: number;
    healthDamage: number;
    health: number;
    attacker: string;
    weapon: string;
    tickNumber: number;
    demoFile: string;

    constructor(victimName: string, armorDamage: number, armor: number,
                healthDamage: number, health: number, attacker: string,
                weapon: string, tickNumber: number, demoFile: string) {
        this.victimName = victimName;
        this.armorDamage = armorDamage;
        this.armor = armor;
        this.healthDamage = healthDamage;
        this.health = health;
        this.attacker = attacker;
        this.weapon = weapon;
        this.tickNumber = tickNumber;
        this.demoFile = demoFile;
        
    }
}

export async function parseHurt(tuple: { value: Uint8Array; done: boolean; }) {
    const linesUnsplit = tuple.value ? utf8Decoder.decode(tuple.value, {stream: true}) : "";
    const lines = linesUnsplit.split("\n");
    for(let lineNumber = 1; lineNumber < lines.length; lineNumber++) {
        if (lines[lineNumber].trim() === "") {
            continue;
        }
        let currentLine = lines[lineNumber].split(",");
        gameData.playerHurt.push(new PlayerHurtRow(
            currentLine[0], parseInt(currentLine[1]), parseInt(currentLine[2]),
            parseInt(currentLine[3]), parseInt(currentLine[4]),
            currentLine[5], currentLine[6], parseInt(currentLine[7]), currentLine[7]
        ));
    }
    if (!tuple.done) {
        await hurtReader.read().then(parseHurt);
    }
}

export let hurtReader:any = null;
export function setHurtReader(readerInput: any) {
    hurtReader = readerInput
}

export class GrenadesRow {
    thrower: string;
    grenadeType: string;
    tickNumber: number;
    demoFile: string;
    
    constructor(thrower: string, grenadeType: string, tickNumber: number, demoFile: string) {
        this.thrower = thrower;
        this.grenadeType = grenadeType;
        this.tickNumber = tickNumber;
        this.demoFile = demoFile;
    }
}

export async function parseGrenades(tuple: { value: Uint8Array; done: boolean; }) {
    const linesUnsplit = tuple.value ? utf8Decoder.decode(tuple.value, {stream: true}) : "";
    const lines = linesUnsplit.split("\n");
    for(let lineNumber = 1; lineNumber < lines.length; lineNumber++) {
        if (lines[lineNumber].trim() === "") {
            continue;
        }
        let currentLine = lines[lineNumber].split(",");
        gameData.grenades.push(new GrenadesRow(
            currentLine[0], currentLine[1], parseInt(currentLine[2]), currentLine[3]
        ));
    }
    if (!tuple.done) {
        await grenadesReader.read().then(parseGrenades);
    }
}

export let grenadesReader:any = null;
export function setGrenadesReader(readerInput: any) {
    grenadesReader = readerInput
}

export class KillsRow {
    killer: string;
    victim: string;
    weapon: string;
    assister: string;
    isHeadshot: boolean;
    isWallbang: boolean;
    penetratedObjects: number;
    tickNumber: number;
    demoFile: string;
    
    constructor(killer: string, victim: string, weapon: string, assister: string,
                isHeadshot: boolean, isWallbang: boolean, penetratedObjects: number,
                tickNumber: number, demoFile: string) {
        this.killer = killer;
        this.victim = victim;
        this.weapon = weapon;
        this.assister = assister;
        this.isHeadshot = isHeadshot;
        this.isWallbang = isWallbang;
        this.penetratedObjects = penetratedObjects;
        this.tickNumber = tickNumber;
        this.demoFile = demoFile;
    }
}

export async function parseKills(tuple: { value: Uint8Array; done: boolean; }) {
    const linesUnsplit = tuple.value ? utf8Decoder.decode(tuple.value, {stream: true}) : "";
    const lines = linesUnsplit.split("\n");
    for(let lineNumber = 1; lineNumber < lines.length; lineNumber++) {
        if (lines[lineNumber].trim() === "") {
            continue;
        }
        let currentLine = lines[lineNumber].split(",");
        gameData.kills.push(new KillsRow(
            currentLine[0], currentLine[1], currentLine[2], currentLine[3],
            parseBool(currentLine[4]), parseBool(currentLine[5]),
            parseInt(currentLine[6]), parseInt(currentLine[7]), currentLine[8]
        ));
    }
    if (!tuple.done) {
        killsReader.read().then(parseKills);
    }
}

export let killsReader:any = null;
export function setKillsReader(readerInput: any) {
    killsReader = readerInput
}

export class GameData {
    position: PositionRow[] = [];
    spotted: SpottedRow[] = [];
    weaponFire: WeaponFireRow[] = [];
    playerHurt: PlayerHurtRow[] = [];
    grenades: GrenadesRow[] = [];
    kills: KillsRow[] = [];
}

export let gameData: GameData = null;
export function createGameData() {
    gameData = new GameData();
}
