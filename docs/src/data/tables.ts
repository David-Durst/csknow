import {gameData} from "./data";
import {InvalidParameterException} from "@aws-sdk/client-cognito-identity";

function parseBool(b: string) {
    return b == "1";
}

export function printTable(keys: string[], values: string[]): string {
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

export class Row {
    id: number;
    foreignKeyValues: number[];
    otherColumnValues: string[];
    parser: Parser;

    constructor(id: number, parser: Parser,
                foreignKeyValues: number[], otherColumnValues: string[]) {
        this.id = id;
        this.parser = parser;
        this.foreignKeyValues = foreignKeyValues;
        this.otherColumnValues = otherColumnValues;
    }

    getHTML(): string {
        return printTable(["id"].concat(this.parser.foreignKeyNames)
                .concat(this.parser.otherColumnNames),
            [this.id.toString()]
                .concat(this.foreignKeyValues.map(((value) => value.toString())))
                .concat(this.otherColumnValues))
    }

    getColumns(): string[] {
        return [this.id.toString()]
            .concat(this.foreignKeyValues.map(((value) => value.toString())))
            .concat(this.otherColumnValues);
    }
}

export class GameRow extends Row {
    demoFile: string;
    demoTickRate: number;

    constructor(id: number, parser: Parser,
                foreignKeyValues: number[], otherColumnValues: string[]) {
        super(id, parser, foreignKeyValues, otherColumnValues);
        this.demoFile = otherColumnValues[0];
        this.demoTickRate = parseInt(otherColumnValues[1]);
    }
}

export class RoundRow extends Row {
    gameId: number;
    startTick: number;
    endTick: number;
    warmup: boolean;
    freezeTimeEnd: number;
    roundNumber: number;
    roundEndReason: number;
    winner: number;
    tWins: number;
    ctWins: number;

    constructor(id: number, parser: Parser,
                foreignKeyValues: number[], otherColumnValues: string[]) {
        super(id, parser, foreignKeyValues, otherColumnValues);
        this.gameId = foreignKeyValues[0];
        this.startTick = parseInt(otherColumnValues[0]);
        this.endTick = parseInt(otherColumnValues[1]);
        this.warmup = parseBool(otherColumnValues[2]);
        this.freezeTimeEnd = parseInt(otherColumnValues[3]);
        this.roundNumber = parseInt(otherColumnValues[4]);
        this.roundEndReason = parseInt(otherColumnValues[5]);
        this.winner = parseInt(otherColumnValues[6]);
        this.tWins = parseInt(otherColumnValues[7]);
        this.ctWins = parseInt(otherColumnValues[8]);
    }
}

export class TickRow extends Row {
    roundId: number;
    gameTime: number;
    demoTickNumber: number;
    gameTickNumber: number;
    bombCarrier: number;
    bombX: number;
    bombY: number;
    bombZ: number;

    constructor(id: number, parser: Parser,
                foreignKeyValues: number[], otherColumnValues: string[]) {
        super(id, parser, foreignKeyValues, otherColumnValues);
        this.roundId = foreignKeyValues[0];
        this.gameTime = parseInt(otherColumnValues[0]);
        this.demoTickNumber = parseInt(otherColumnValues[1]);
        this.gameTickNumber = parseInt(otherColumnValues[2]);
        this.bombCarrier = parseInt(otherColumnValues[3]);
        this.bombX = parseFloat(otherColumnValues[4]);
        this.bombY = parseFloat(otherColumnValues[5]);
        this.bombZ = parseFloat(otherColumnValues[6]);
    }
}

export class PlayerRow extends Row {
    name: string;

    constructor(id: number, parser: Parser,
                foreignKeyValues: number[], otherColumnValues: string[]) {
        super(id, parser, foreignKeyValues, otherColumnValues);
        this.name = otherColumnValues[0];
    }
}

export class PlayerAtTickRow extends Row {
    playerId: number;
    tickId: number;
    posX: number;
    posY: number;
    posZ: number;
    viewX: number;
    viewY: number;
    team: number;
    health: number;
    armor: number;
    isAlive: boolean;

    constructor(id: number, parser: Parser,
                foreignKeyValues: number[], otherColumnValues: string[]) {
        super(id, parser, foreignKeyValues, otherColumnValues);
        this.playerId = parseInt(otherColumnValues[0]);
        this.tickId = foreignKeyValues[0];
        this.posX = parseFloat(otherColumnValues[1]);
        this.posY = parseFloat(otherColumnValues[2]);
        this.posZ = parseFloat(otherColumnValues[3]);
        this.viewX = parseFloat(otherColumnValues[4]);
        this.viewY = parseFloat(otherColumnValues[5]);
        this.team = parseInt(otherColumnValues[6]);
        this.health = parseFloat(otherColumnValues[7]);
        this.armor = parseFloat(otherColumnValues[8]);
        this.isAlive = parseBool(otherColumnValues[9]);
    }
}

export enum ParserType {
    tick,
    game,
    round,
    player,
    playerAtTick,
    other
}

function getOtherColumnsStart(foreignKeyStart: number, foreignKeyLength: number) {
    if (foreignKeyLength > 0) {
        return foreignKeyStart + foreignKeyLength + 1;
    }
    else {
        return foreignKeyStart;
    }
}

export class Parser {
    tempLineContainer: string = "";
    tableName: string
    foreignKeyNames: string[]
    otherColumnNames: string[];
    keyPlayerColumns: number[];
    // if variable number of ticks per event (variableLength), set ticks column
    // otherwise set ticksPerEvent
    variableLength: boolean;
    ticksPerEvent: number;
    ticksColumn: number;
    parserType: ParserType;

    constructor(tableName: string, foreignKeyNames: string[],
                otherColumnNames: string[],
                ticksPerEvent: string, parserType: ParserType) {
        this.tableName = tableName;
        this.foreignKeyNames = foreignKeyNames;
        this.otherColumnNames = otherColumnNames;
        if (ticksPerEvent[0] == 'c') {
            this.variableLength = true;
            this.ticksPerEvent = -1;
            this.ticksColumn = parseInt(ticksPerEvent.slice(1));
        }
        else {
            this.variableLength = false;
            this.ticksPerEvent = parseInt(ticksPerEvent);
            this.ticksColumn = -1;
        }
        this.parserType = parserType;
    }

    parseOneLine(currentLine: string[]) {
        const id = parseInt(currentLine[0]);
        const foreignKeysStart = 1;
        const otherColumnsStart = getOtherColumnsStart(foreignKeysStart,
            this.foreignKeyNames.length);
        if (this.parserType == ParserType.tick) {
            gameData.ticksTable.push(
                new TickRow(
                    id, this,
                    currentLine.slice(foreignKeysStart,
                        foreignKeysStart + this.foreignKeyNames.length)
                        .map(parseInt),
                    currentLine.slice(otherColumnsStart, currentLine.length)
                )
            )
        }
        else if (this.parserType == ParserType.game) {
            gameData.gamesTable.push(
                new GameRow(
                    id, this,
                    currentLine.slice(foreignKeysStart,
                        foreignKeysStart + this.foreignKeyNames.length)
                        .map(parseInt),
                    currentLine.slice(otherColumnsStart, currentLine.length)
                )
            )
        }
        else if (this.parserType == ParserType.round) {
            gameData.roundsTable.push(
                new RoundRow(
                    id, this,
                    currentLine.slice(foreignKeysStart,
                        foreignKeysStart + this.foreignKeyNames.length)
                        .map(parseInt),
                    currentLine.slice(otherColumnsStart, currentLine.length)
                )
            )
        }
        else if (this.parserType == ParserType.player) {
            gameData.playersTable.push(
                new PlayerRow(
                    id, this,
                    currentLine.slice(foreignKeysStart,
                        foreignKeysStart + this.foreignKeyNames.length)
                        .map(parseInt),
                    currentLine.slice(otherColumnsStart, currentLine.length)
                )
            )
        }
        else if (this.parserType == ParserType.playerAtTick) {
            gameData.playerAtTicksTable.push(
                new PlayerAtTickRow(
                    id, this,
                    currentLine.slice(foreignKeysStart,
                        foreignKeysStart + this.foreignKeyNames.length)
                        .map(parseInt),
                    currentLine.slice(otherColumnsStart, currentLine.length)
                )
            )
        }
        else {
            gameData.tables.get(this.tableName).push(
                new Row(
                    id, this,
                    currentLine.slice(foreignKeysStart,
                        foreignKeysStart + this.foreignKeyNames.length)
                        .map(parseInt),
                    currentLine.slice(otherColumnsStart, currentLine.length)
                )
            )
        }
    }

    reader: any = null
    setReader(readerInput: any) {
        this.reader = readerInput
    }

}

export type Index = Map<number, number[]>;
export const tickTableName = "ticks"
export const gameTableName = "games"
export const roundTableName = "rounds"
export const playerAtTickTableName = "playerAtTick"
export const playersTableName = "players"
export class GameData {
    tableNames: string[] = [];
    parsers: Map<string, Parser> = new Map<string, Parser>();
    gamesTable: GameRow[];
    roundsTable: RoundRow[];
    roundIdToIndex: Map<number, number> = new Map<number, number>();
    ticksTable: TickRow[];
    playersTable: PlayerRow[];
    playerIdToIndex: Map<number, number> = new Map<number, number>();
    playerAtTicksTable: PlayerAtTickRow[];
    tables: Map<string, Row[]> =
        new Map<string, Row[]>();
    ticksToOtherTablesIndices: Map<string, Index> =
        new Map<string, Map<number, number[]>>();

    getRound(tickData: TickRow) : RoundRow {
        if (this.roundIdToIndex.size == 0) {
            for (let i = 0; i < this.roundsTable.length; i++) {
                this.roundIdToIndex.set(this.roundsTable[i].id, i);
            }
        }
        return this.roundsTable[this.roundIdToIndex.get(tickData.roundId)]
    }

    getPlayers(tickData: TickRow) {
        return this.ticksToOtherTablesIndices
            .get(playerAtTickTableName).get(tickData.id)
            .map((value) => gameData.playerAtTicksTable[value])
    }

    getPlayerIndex(playerId: number) : number {
        if (this.playerIdToIndex.size == 0) {
            for (let i = 0; i < this.playersTable.length; i++) {
                this.playerIdToIndex.set(this.playersTable[i].id, i);
            }
        }
        return this.playerIdToIndex.get(playerId);
    }

    getPlayerName(playerId: number) : string {
        if (this.playerIdToIndex.size == 0) {
            for (let i = 0; i < this.playersTable.length; i++) {
                this.playerIdToIndex.set(this.playersTable[i].id, i);
            }
        }
        return this.playersTable[this.playerIdToIndex.get(playerId)].name
    }

    clone(target: GameData) {
        target.tableNames = this.tableNames;
        target.parsers = this.parsers;
        target.roundsTable = this.roundsTable;
        target.roundIdToIndex = this.roundIdToIndex
        target.ticksTable = this.ticksTable;
        target.playersTable = this.playersTable
        target.playerIdToIndex = this.playerIdToIndex
        target.playerAtTicksTable = this.playerAtTicksTable;
        target.tables = this.tables;
        target.ticksToOtherTablesIndices = this.ticksToOtherTablesIndices;
    }
}

export function getTickToOtherTableIndex(gameData: GameData, tableName: string):
    Map<number, number[]> {
    if (tableName == "none") {
        return null
    }
    else if (tableName == playerAtTickTableName || gameData.tableNames.includes(tableName)) {
        return gameData.ticksToOtherTablesIndices.get(tableName)
    }
    else {
        throw new Error("getEventIndex for invalid tableName string " + tableName)
    }
}