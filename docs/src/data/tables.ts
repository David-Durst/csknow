import {gameData} from "./data";
import {InvalidParameterException} from "@aws-sdk/client-cognito-identity";

function parseBool(b: string) {
    return b == "1";
}

export interface Parseable {
    tempLineContainer: string;

    parseOneLine(currentLine: string[]): any

    reader: ReadableStreamDefaultReader<any>
}

export function setReader(readerInput: any, container: Parseable) {
    container.reader = readerInput
}

export interface DemoData {
    demoTickNumber: number
    demoFile: string
}

export interface Printable {
    getHTML(): string
    getSource?(): string
    getTargets?(): string[]
}

function printTable(keys: string[], values: string[]): string {
    let result = "<div>"
    for (let i = 0; i < keys.length; i++) {
        if (i == 0) {
            result += "<span class='entry first-entry'>"
        }
        else {
            result += "<span class='entry'>"
        }
        result += "<span class='label'>" + keys[i] + ":</span>" +
            "<span id='t_score'>" + values[i] + "</span></span>"
    }
    return result + "</div>"
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
    isBlinded: boolean;

    constructor(name: string, team: number, xPosition: number, yPosition: number,
                zPosition: number, xViewDirection: number, yViewDirection: number,
                isAlive: boolean, isBlinded: boolean) {
        this.name = name
        this.team = team
        this.xPosition = xPosition
        this.yPosition = yPosition
        this.zPosition = zPosition
        this.xViewDirection = xViewDirection
        this.yViewDirection = yViewDirection
        this.isAlive = isAlive
        this.isBlinded = isBlinded
    }
}

export class PositionRow implements DemoData {
    demoTickNumber: number;
    gameTickNumber: number;
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

    constructor(demoTickNumber: number, gameTickNumber: number, matchStarted: boolean, gamePhase: number,
                roundsPlayed: number, isWarmup: boolean, roundStart: boolean, roundEnd: boolean,
                roundEndReason: number, freezeTimeEnded: boolean, tScore: number, ctScore: number, numPlayers: number,
                player0Name: string, player0Team: number, player0XPosition: number, player0YPosition: number,
                player0ZPosition: number, player0XViewDirection: number, player0YViewDirection: number,
                player0IsAlive: boolean, player0isBlinded: boolean,
                player1Name: string, player1Team: number, player1XPosition: number, player1YPosition: number,
                player1ZPosition: number, player1XViewDirection: number, player1YViewDirection: number,
                player1IsAlive: boolean, player1isBlinded: boolean,
                player2Name: string, player2Team: number, player2XPosition: number, player2YPosition: number,
                player2ZPosition: number, player2XViewDirection: number, player2YViewDirection: number,
                player2IsAlive: boolean, player2isBlinded: boolean,
                player3Name: string, player3Team: number, player3XPosition: number, player3YPosition: number,
                player3ZPosition: number, player3XViewDirection: number, player3YViewDirection: number,
                player3IsAlive: boolean, player3isBlinded: boolean,
                player4Name: string, player4Team: number, player4XPosition: number, player4YPosition: number,
                player4ZPosition: number, player4XViewDirection: number, player4YViewDirection: number,
                player4IsAlive: boolean, player4isBlinded: boolean,
                player5Name: string, player5Team: number, player5XPosition: number, player5YPosition: number,
                player5ZPosition: number, player5XViewDirection: number, player5YViewDirection: number,
                player5IsAlive: boolean, player5isBlinded: boolean,
                player6Name: string, player6Team: number, player6XPosition: number, player6YPosition: number,
                player6ZPosition: number, player6XViewDirection: number, player6YViewDirection: number,
                player6IsAlive: boolean, player6isBlinded: boolean,
                player7Name: string, player7Team: number, player7XPosition: number, player7YPosition: number,
                player7ZPosition: number, player7XViewDirection: number, player7YViewDirection: number,
                player7IsAlive: boolean, player7isBlinded: boolean,
                player8Name: string, player8Team: number, player8XPosition: number, player8YPosition: number,
                player8ZPosition: number, player8XViewDirection: number, player8YViewDirection: number,
                player8IsAlive: boolean, player8isBlinded: boolean,
                player9Name: string, player9Team: number, player9XPosition: number, player9YPosition: number,
                player9ZPosition: number, player9XViewDirection: number, player9YViewDirection: number,
                player9IsAlive: boolean, player9isBlinded: boolean,
                demoFile: string
    ) {
        this.demoTickNumber = demoTickNumber;
        this.gameTickNumber = gameTickNumber;
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
            player0isBlinded))
        this.players.push(new PlayerPositionRow(
            player1Name, player1Team, player1XPosition, player1YPosition,
            player1ZPosition, player1XViewDirection, player1YViewDirection, player1IsAlive,
            player1isBlinded))
        this.players.push(new PlayerPositionRow(
            player2Name, player2Team, player2XPosition, player2YPosition,
            player2ZPosition, player2XViewDirection, player2YViewDirection, player2IsAlive,
            player2isBlinded))
        this.players.push(new PlayerPositionRow(
            player3Name, player3Team, player3XPosition, player3YPosition,
            player3ZPosition, player3XViewDirection, player3YViewDirection, player3IsAlive,
            player3isBlinded))
        this.players.push(new PlayerPositionRow(
            player4Name, player4Team, player4XPosition, player4YPosition,
            player4ZPosition, player4XViewDirection, player4YViewDirection, player4IsAlive,
            player4isBlinded))
        this.players.push(new PlayerPositionRow(
            player5Name, player5Team, player5XPosition, player5YPosition,
            player5ZPosition, player5XViewDirection, player5YViewDirection, player5IsAlive,
            player5isBlinded))
        this.players.push(new PlayerPositionRow(
            player6Name, player6Team, player6XPosition, player6YPosition,
            player6ZPosition, player6XViewDirection, player6YViewDirection, player6IsAlive,
            player6isBlinded))
        this.players.push(new PlayerPositionRow(
            player7Name, player7Team, player7XPosition, player7YPosition,
            player7ZPosition, player7XViewDirection, player7YViewDirection, player7IsAlive,
            player7isBlinded))
        this.players.push(new PlayerPositionRow(
            player8Name, player8Team, player8XPosition, player8YPosition,
            player8ZPosition, player8XViewDirection, player8YViewDirection, player8IsAlive,
            player8isBlinded))
        this.players.push(new PlayerPositionRow(
            player9Name, player9Team, player9XPosition, player9YPosition,
            player9ZPosition, player9XViewDirection, player9YViewDirection, player9IsAlive,
            player9isBlinded))
        this.demoFile = demoFile;
    }
}

export class PositionParser implements Parseable {
    tempLineContainer: string = "";

    parseOneLine(currentLine: string[]): any {
        // skip warmup
        if (parseBool(currentLine[5])) {
            return;
        }
        if (isNaN(parseInt(currentLine[0]))) {
            console.log(currentLine)
        }
        gameData.position.push(new PositionRow(
            // first 12 aren't player specified
            parseInt(currentLine[0]), parseInt(currentLine[1]), parseBool(currentLine[2]), parseInt(currentLine[3]),
            parseInt(currentLine[4]), parseBool(currentLine[5]), parseBool(currentLine[6]),
            parseBool(currentLine[7]), parseInt(currentLine[8]), parseBool(currentLine[9]),
            parseInt(currentLine[10]), parseInt(currentLine[11]), parseInt(currentLine[12]),
            // each player is 10 entries
            // player 0
            currentLine[13], parseInt(currentLine[14]), parseFloat(currentLine[15]),
            parseFloat(currentLine[16]), parseFloat(currentLine[17]), parseFloat(currentLine[18]),
            parseFloat(currentLine[19]), parseBool(currentLine[20]), parseBool(currentLine[21]),
            // player 1
            currentLine[22], parseInt(currentLine[23]), parseFloat(currentLine[24]),
            parseFloat(currentLine[25]), parseFloat(currentLine[26]), parseFloat(currentLine[27]),
            parseFloat(currentLine[28]), parseBool(currentLine[29]), parseBool(currentLine[30]),
            // player 2
            currentLine[31], parseInt(currentLine[32]), parseFloat(currentLine[33]),
            parseFloat(currentLine[34]), parseFloat(currentLine[35]), parseFloat(currentLine[36]),
            parseFloat(currentLine[37]), parseBool(currentLine[38]), parseBool(currentLine[39]),
            // player 3
            currentLine[40], parseInt(currentLine[41]), parseFloat(currentLine[42]),
            parseFloat(currentLine[43]), parseFloat(currentLine[44]), parseFloat(currentLine[45]),
            parseFloat(currentLine[46]), parseBool(currentLine[47]), parseBool(currentLine[48]),
            // player 4
            currentLine[49], parseInt(currentLine[50]), parseFloat(currentLine[51]),
            parseFloat(currentLine[52]), parseFloat(currentLine[53]), parseFloat(currentLine[54]),
            parseFloat(currentLine[55]), parseBool(currentLine[56]), parseBool(currentLine[57]),
            // player 5
            currentLine[58], parseInt(currentLine[59]), parseFloat(currentLine[60]),
            parseFloat(currentLine[61]), parseFloat(currentLine[62]), parseFloat(currentLine[63]),
            parseFloat(currentLine[64]), parseBool(currentLine[65]), parseBool(currentLine[66]),
            // player 6
            currentLine[67], parseInt(currentLine[68]), parseFloat(currentLine[69]),
            parseFloat(currentLine[70]), parseFloat(currentLine[71]), parseFloat(currentLine[72]),
            parseFloat(currentLine[73]), parseBool(currentLine[74]), parseBool(currentLine[75]),
            // player 7
            currentLine[76], parseInt(currentLine[77]), parseFloat(currentLine[78]),
            parseFloat(currentLine[79]), parseFloat(currentLine[80]), parseFloat(currentLine[81]),
            parseFloat(currentLine[82]), parseBool(currentLine[83]), parseBool(currentLine[84]),
            // player 8
            currentLine[85], parseInt(currentLine[86]), parseFloat(currentLine[87]),
            parseFloat(currentLine[88]), parseFloat(currentLine[89]), parseFloat(currentLine[90]),
            parseFloat(currentLine[91]), parseBool(currentLine[92]), parseBool(currentLine[93]),
            // player 9
            currentLine[94], parseInt(currentLine[95]), parseFloat(currentLine[96]),
            parseFloat(currentLine[97]), parseFloat(currentLine[98]), parseFloat(currentLine[99]),
            parseFloat(currentLine[100]), parseBool(currentLine[101]), parseBool(currentLine[102]),
            // after player data
            currentLine[103]
        ));
    }

    reader: any = null;
}

export class SpottedRow implements DemoData, Printable{
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
    demoTickNumber: number;
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
        this.demoTickNumber = tickNumber;
        this.demoFile = demoFile;
    }

    getHTML(): string {
        let keys: string[] = ["demo tick", "spotted player",
            this.player0Name, this.player1Name, this.player2Name, this.player3Name,
            this.player4Name, this.player5Name, this.player6Name, this.player7Name,
            this.player8Name, this.player9Name]
        let values: string[] = [this.demoTickNumber.toString(), this.spottedPlayer,
            String(this.player0Spotter), String(this.player1Spotter), String(this.player2Spotter),
            String(this.player3Spotter), String(this.player4Spotter), String(this.player5Spotter),
            String(this.player6Spotter), String(this.player7Spotter), String(this.player8Spotter),
            String(this.player9Spotter)]
        return printTable(keys, values)
    }
}

export class SpottedParser implements Parseable {
    tempLineContainer: string = "";

    parseOneLine(currentLine: string[]): any {
        gameData.spotted.push(new SpottedRow(
            currentLine[0], currentLine[1], parseBool(currentLine[2]),
            currentLine[3], parseBool(currentLine[4]), currentLine[5], parseBool(currentLine[6]),
            currentLine[7], parseBool(currentLine[8]), currentLine[9], parseBool(currentLine[10]),
            currentLine[11], parseBool(currentLine[12]), currentLine[13], parseBool(currentLine[13]),
            currentLine[14], parseBool(currentLine[15]), currentLine[16], parseBool(currentLine[18]),
            currentLine[19], parseBool(currentLine[20]), parseInt(currentLine[21]), currentLine[22]
        ));
    }

    reader: any = null
}

export class WeaponFireRow implements DemoData, Printable {
    shooter: string;
    weapon: string;
    demoTickNumber: number;
    demoFile: string;

    constructor(shooter: string, weapon: string,
                tickNumber: number, demoFile: string) {
        this.shooter = shooter;
        this.weapon = weapon;
        this.demoTickNumber = tickNumber;
        this.demoFile = demoFile;
    }

    getHTML(): string {
        return printTable(["demo tick", "shooter", "weapon"],
            [this.demoTickNumber.toString(), this.shooter, this.weapon])
    }

    getSource(): string {
        return this.shooter
    }
}

export class WeaponFireParser implements Parseable {
    tempLineContainer: string = "";

    parseOneLine(currentLine: string[]): any {
        gameData.weaponFire.push(new WeaponFireRow(
            currentLine[0], currentLine[1], parseInt(currentLine[2]), currentLine[3]
        ));
    }

    reader: any = null
}

export class PlayerHurtRow implements DemoData, Printable {
    victimName: string;
    armorDamage: number;
    armor: number;
    healthDamage: number;
    health: number;
    attacker: string;
    weapon: string;
    demoTickNumber: number;
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
        this.demoTickNumber = tickNumber;
        this.demoFile = demoFile;

    }

    getHTML(): string {
        return printTable(["demo tick", "victim", "armor", "armor damage",
                "health", "health damage", "attacker", "weapon"],
            [this.demoTickNumber.toString(), this.victimName, this.armor.toString(),
                this.armorDamage.toString(), this.health.toString(), this.healthDamage.toString(),
                this.attacker, this.weapon])
    }

    getSource(): string {
        return this.victimName
    }

    getTargets(): string[] {
        return [this.attacker]
    }
}

export class PlayerHurtParser implements Parseable {
    tempLineContainer: string = "";

    parseOneLine(currentLine: string[]): any {
        gameData.playerHurt.push(new PlayerHurtRow(
            currentLine[0], parseInt(currentLine[1]), parseInt(currentLine[2]),
            parseInt(currentLine[3]), parseInt(currentLine[4]),
            currentLine[5], currentLine[6], parseInt(currentLine[7]), currentLine[8]
        ));
    }

    reader: any = null;
}

export class GrenadesRow implements DemoData, Printable {
    thrower: string;
    grenadeType: string;
    demoTickNumber: number;
    demoFile: string;

    constructor(thrower: string, grenadeType: string, tickNumber: number, demoFile: string) {
        this.thrower = thrower;
        this.grenadeType = grenadeType;
        this.demoTickNumber = tickNumber;
        this.demoFile = demoFile;
    }

    getHTML(): string {
        return printTable(["demo tick", "thrower", "grenade type"],
            [this.demoTickNumber.toString(), this.thrower, this.grenadeType])
    }

    getSource(): string {
        return this.thrower
    }
}

export class GrenadesParser implements Parseable {
    tempLineContainer: string = "";

    parseOneLine(currentLine: string[]): any {
        gameData.grenades.push(new GrenadesRow(
            currentLine[0], currentLine[1], parseInt(currentLine[2]), currentLine[3]
        ));
    }

    reader: any = null
}

export class KillsRow implements DemoData, Printable {
    killer: string;
    victim: string;
    weapon: string;
    assister: string;
    isHeadshot: boolean;
    isWallbang: boolean;
    penetratedObjects: number;
    demoTickNumber: number;
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
        this.demoTickNumber = tickNumber;
        this.demoFile = demoFile;
    }

    getHTML(): string {
        return printTable(["demo tick", "killer", "victim", "weapon",
                "assister", "headshot", "wallbang", "penetrated objects"],
            [this.demoTickNumber.toString(), this.killer, this.victim,
                this.weapon, this.assister, String(this.isHeadshot),
                String(this.isWallbang), this.penetratedObjects.toString()])
    }

    getSource(): string {
        return this.killer
    }

    getTargets(): string[] {
        let results = [this.killer]
        if (this.assister != "n/a") {
            results.push(this.assister)
        }
        return results
    }
}

export class KillsParser implements Parseable {
    tempLineContainer: string = "";

    parseOneLine(currentLine: string[]): any {
        gameData.kills.push(new KillsRow(
            currentLine[0], currentLine[1], currentLine[2], currentLine[3],
            parseBool(currentLine[4]), parseBool(currentLine[5]),
            parseInt(currentLine[6]), parseInt(currentLine[7]), currentLine[8]
        ));
    }

    reader: any = null
}

export class GameData {
    positionParser: PositionParser = new PositionParser();
    position: PositionRow[] = [];
    spottedParser: SpottedParser = new SpottedParser();
    spotted: SpottedRow[] = [];
    positionToSpotted: Map<number, number[]> = new Map<number, number[]>()
    weaponFireParser: WeaponFireParser = new WeaponFireParser();
    weaponFire: WeaponFireRow[] = [];
    positionToWeaponFire: Map<number, number[]> = new Map<number, number[]>()
    playerHurtParser: PlayerHurtParser = new PlayerHurtParser();
    playerHurt: PlayerHurtRow[] = [];
    positionToPlayerHurt: Map<number, number[]> = new Map<number, number[]>()
    grenadeParser: GrenadesParser = new GrenadesParser();
    grenades: GrenadesRow[] = [];
    positionToGrenades: Map<number, number[]> = new Map<number, number[]>()
    killsParser: KillsParser = new KillsParser();
    kills: KillsRow[] = [];
    positionToKills: Map<number, number[]> = new Map<number, number[]>()
    
    clone(target: GameData) {
        target.positionParser = this.positionParser
        target.position = this.position
        target.spottedParser = this.spottedParser
        target.spotted = this.spotted
        target.positionToSpotted = this.positionToSpotted
        target.weaponFireParser = this.weaponFireParser
        target.weaponFire = this.weaponFire
        target.positionToWeaponFire = this.positionToWeaponFire
        target.playerHurtParser = this.playerHurtParser
        target.playerHurt = this.playerHurt
        target.positionToPlayerHurt = this.positionToPlayerHurt
        target.grenadeParser = this.grenadeParser
        target.grenades = this.grenades
        target.positionToGrenades = this.positionToGrenades
        target.killsParser = this.killsParser
        target.kills = this.kills
        target.positionToKills = this.positionToKills
    }
}

export function getEventIndex(gameData: GameData, event: string): Map<number, number[]> {
    if (event == "spotted") {
        return gameData.positionToSpotted
    }
    else if (event == "weapon_fire") {
        return gameData.positionToWeaponFire
    }
    else if (event == "hurt") {
        return gameData.positionToPlayerHurt
    }
    else if (event == "grenades") {
        return gameData.positionToGrenades
    }
    else if (event == "kills") {
        return gameData.positionToKills
    }
    else {
        throw new Error("getEventIndex for invalid event string " + event)
    }
}

export function getEventArray(gameData: GameData, event: string): Printable[] {
    if (event == "spotted") {
        return gameData.spotted
    }
    else if (event == "weapon_fire") {
        return gameData.weaponFire
    }
    else if (event == "hurt") {
        return gameData.playerHurt
    }
    else if (event == "grenades") {
        return gameData.grenades
    }
    else if (event == "kills") {
        return gameData.grenades
    }
    else {
        throw new Error("getEventIndex for invalid event string " + event)
    }
}