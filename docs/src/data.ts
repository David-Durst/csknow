const utf8Decoder = new TextDecoder("utf-8");

function parseBool(b: string) {
    return b == "1";
}

class PlayerPositionRow {
    name: string;
    team: number;
    xPosition: number;
    yPosition: number;
    zPosition: number;
    xViewDirection: number;
    yViewDirection: number;
    isAlive: boolean;
    // remove this later when fix data set
    teamRepeat: number;
    isBlinded: boolean;
    
    constructor(name: string, team: number, xPosition: number, yPosition: number,
            zPosition: number, xViewDirection: number, yViewDirection: number,
            isAlive: boolean, teamRepeat: number, isBlinded: boolean) {
        this.name = name
        this.team = team
        this.xPosition = xPosition
        this.yPosition = yPosition
        this.zPosition = zPosition
        this.xViewDirection = xViewDirection
        this.yViewDirection = yViewDirection
        this.isAlive = isAlive
        this.teamRepeat = teamRepeat
        this.isBlinded = isBlinded
    }
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
    players: PlayerPositionRow[];
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
        this.players = []
        for (let i = 0; i < 10; i++) {
        }
        this.players.push(new PlayerPositionRow(
            player0Name, player0Team, player0XPosition, player0YPosition,
            player0ZPosition, player0XViewDirection, player0YViewDirection, player0IsAlive,
            // remove this later when fix data set
            player0TeamRepeat, player0isBlinded))
        this.players.push(new PlayerPositionRow(
            player1Name, player1Team, player1XPosition, player1YPosition,
            player1ZPosition, player1XViewDirection, player1YViewDirection, player1IsAlive,
            // remove this later when fix data set
            player1TeamRepeat, player1isBlinded))
        this.players.push(new PlayerPositionRow(
            player2Name, player2Team, player2XPosition, player2YPosition,
            player2ZPosition, player2XViewDirection, player2YViewDirection, player2IsAlive,
            // remove this later when fix data set
            player2TeamRepeat, player2isBlinded))
        this.players.push(new PlayerPositionRow(
            player3Name, player3Team, player3XPosition, player3YPosition,
            player3ZPosition, player3XViewDirection, player3YViewDirection, player3IsAlive,
            // remove this later when fix data set
            player3TeamRepeat, player3isBlinded))
        this.players.push(new PlayerPositionRow(
            player4Name, player4Team, player4XPosition, player4YPosition,
            player4ZPosition, player4XViewDirection, player4YViewDirection, player4IsAlive,
            // remove this later when fix data set
            player4TeamRepeat, player4isBlinded))
        this.players.push(new PlayerPositionRow(
            player5Name, player5Team, player5XPosition, player5YPosition,
            player5ZPosition, player5XViewDirection, player5YViewDirection, player5IsAlive,
            // remove this later when fix data set
            player5TeamRepeat, player5isBlinded))
        this.players.push(new PlayerPositionRow(
            player6Name, player6Team, player6XPosition, player6YPosition,
            player6ZPosition, player6XViewDirection, player6YViewDirection, player6IsAlive,
            // remove this later when fix data set
            player6TeamRepeat, player6isBlinded))
        this.players.push(new PlayerPositionRow(
            player7Name, player7Team, player7XPosition, player7YPosition,
            player7ZPosition, player7XViewDirection, player7YViewDirection, player7IsAlive,
            // remove this later when fix data set
            player7TeamRepeat, player7isBlinded))
        this.players.push(new PlayerPositionRow(
            player8Name, player8Team, player8XPosition, player8YPosition,
            player8ZPosition, player8XViewDirection, player8YViewDirection, player8IsAlive,
            // remove this later when fix data set
            player8TeamRepeat, player8isBlinded))
        this.players.push(new PlayerPositionRow(
            player9Name, player9Team, player9XPosition, player9YPosition,
            player9ZPosition, player9XViewDirection, player9YViewDirection, player9IsAlive,
            // remove this later when fix data set
            player9TeamRepeat, player9isBlinded))
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
        // skip warmup
        if (parseBool(currentLine[4])) {
            continue;
        }
        gameData.position.push(new PositionRow(
            // first 12 aren't palyer specified
            parseInt(currentLine[0]), parseBool(currentLine[1]), parseInt(currentLine[2]),
            parseInt(currentLine[3]), parseBool(currentLine[4]), parseBool(currentLine[5]),
            parseBool(currentLine[6]), parseInt(currentLine[7]), parseBool(currentLine[8]),
            parseInt(currentLine[9]), parseInt(currentLine[10]), parseInt(currentLine[11]),
            // each player is 10 entries
            // player 0
            currentLine[12], parseInt(currentLine[13]), parseFloat(currentLine[14]),
            parseFloat(currentLine[15]), parseFloat(currentLine[16]), parseFloat(currentLine[17]),
            parseFloat(currentLine[18]), parseBool(currentLine[19]), parseInt(currentLine[20]), parseBool(currentLine[21]),
            // player 1
            currentLine[22], parseInt(currentLine[23]), parseFloat(currentLine[24]),
            parseFloat(currentLine[25]), parseFloat(currentLine[26]), parseFloat(currentLine[27]),
            parseFloat(currentLine[28]), parseBool(currentLine[29]), parseInt(currentLine[30]), parseBool(currentLine[31]),
            // player 2
            currentLine[32], parseInt(currentLine[33]), parseFloat(currentLine[34]),
            parseFloat(currentLine[35]), parseFloat(currentLine[36]), parseFloat(currentLine[37]),
            parseFloat(currentLine[38]), parseBool(currentLine[39]), parseInt(currentLine[40]), parseBool(currentLine[41]),
            // player 3
            currentLine[42], parseInt(currentLine[43]), parseFloat(currentLine[44]),
            parseFloat(currentLine[45]), parseFloat(currentLine[46]), parseFloat(currentLine[47]),
            parseFloat(currentLine[48]), parseBool(currentLine[49]), parseInt(currentLine[50]), parseBool(currentLine[51]),
            // player 4
            currentLine[52], parseInt(currentLine[53]), parseFloat(currentLine[54]),
            parseFloat(currentLine[55]), parseFloat(currentLine[56]), parseFloat(currentLine[57]),
            parseFloat(currentLine[58]), parseBool(currentLine[59]), parseInt(currentLine[60]), parseBool(currentLine[61]),
            // player 5
            currentLine[62], parseInt(currentLine[63]), parseFloat(currentLine[64]),
            parseFloat(currentLine[65]), parseFloat(currentLine[66]), parseFloat(currentLine[67]),
            parseFloat(currentLine[68]), parseBool(currentLine[69]), parseInt(currentLine[70]), parseBool(currentLine[71]),
            // player 6
            currentLine[72], parseInt(currentLine[73]), parseFloat(currentLine[74]),
            parseFloat(currentLine[75]), parseFloat(currentLine[76]), parseFloat(currentLine[77]),
            parseFloat(currentLine[78]), parseBool(currentLine[79]), parseInt(currentLine[80]), parseBool(currentLine[81]),
            // player 7
            currentLine[82], parseInt(currentLine[83]), parseFloat(currentLine[84]),
            parseFloat(currentLine[85]), parseFloat(currentLine[86]), parseFloat(currentLine[87]),
            parseFloat(currentLine[88]), parseBool(currentLine[89]), parseInt(currentLine[90]), parseBool(currentLine[91]),
            // player 8
            currentLine[92], parseInt(currentLine[93]), parseFloat(currentLine[94]),
            parseFloat(currentLine[95]), parseFloat(currentLine[96]), parseFloat(currentLine[97]),
            parseFloat(currentLine[98]), parseBool(currentLine[99]), parseInt(currentLine[100]), parseBool(currentLine[101]),
            // player 9
            currentLine[102], parseInt(currentLine[103]), parseFloat(currentLine[104]),
            parseFloat(currentLine[105]), parseFloat(currentLine[106]), parseFloat(currentLine[107]),
            parseFloat(currentLine[108]), parseBool(currentLine[109]), parseInt(currentLine[110]), parseBool(currentLine[111]),
            // after player data
            currentLine[112]
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
