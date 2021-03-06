import {gameData} from "./data";
import IntervalTree from "@flatten-js/interval-tree";

export function parseBool(b: string) {
    return b == "1" || b == "true";
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

    getStartTick(): number {
        return this.foreignKeyValues[this.parser.startTickColumn]
    }
}

export class GameRow extends Row {
    demoFile: string;
    demoTickRate: number;
    gameTickRate: number;

    constructor(id: number, parser: Parser,
                foreignKeyValues: number[], otherColumnValues: string[]) {
        super(id, parser, foreignKeyValues, otherColumnValues);
        this.demoFile = otherColumnValues[0];
        this.demoTickRate = parseInt(otherColumnValues[1]);
        this.gameTickRate = parseInt(otherColumnValues[2]);
    }
}

export class RoundRow extends Row {
    gameId: number;
    startTick: number;
    endTick: number;
    roundLength: number;
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
        this.startTick = parseInt(otherColumnValues[1]);
        this.endTick = parseInt(otherColumnValues[2]);
        this.roundLength = parseInt(otherColumnValues[3]);
        this.warmup = parseBool(otherColumnValues[0]);
        this.freezeTimeEnd = parseInt(otherColumnValues[1]);
        this.roundNumber = parseInt(otherColumnValues[2]);
        this.roundEndReason = parseInt(otherColumnValues[3]);
        this.winner = parseInt(otherColumnValues[4]);
        this.tWins = parseInt(otherColumnValues[5]);
        this.ctWins = parseInt(otherColumnValues[6]);
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
    gameId: number;
    name: string;

    constructor(id: number, parser: Parser,
                foreignKeyValues: number[], otherColumnValues: string[]) {
        super(id, parser, foreignKeyValues, otherColumnValues);
        this.gameId = foreignKeyValues[0];
        this.name = otherColumnValues[0];
    }
}

export class PlayerAtTickRow extends Row {
    tickId: number;
    playerId: number;
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
        this.tickId = foreignKeyValues[0];
        this.playerId = foreignKeyValues[1];
        this.posX = parseFloat(otherColumnValues[0]);
        this.posY = parseFloat(otherColumnValues[1]);
        this.posZ = parseFloat(otherColumnValues[2]);
        this.viewX = parseFloat(otherColumnValues[3]);
        this.viewY = parseFloat(otherColumnValues[4]);
        this.team = parseInt(otherColumnValues[5]);
        this.health = parseFloat(otherColumnValues[6]);
        this.armor = parseFloat(otherColumnValues[7]);
        this.isAlive = parseBool(otherColumnValues[8]);
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

export class Parser {
    tempLineContainer: string = "";
    tableName: string
    foreignKeyNames: string[]
    otherColumnNames: string[];
    keyPlayerColumns: number[];
    startTickColumn: number;
    // if variable number of ticks per event (variableLength), set ticks column
    // otherwise set ticksPerEvent
    variableLength: boolean;
    ticksPerEvent: number;
    ticksColumn: number;
    parserType: ParserType;
    baseUrl: string;
    filterUrl: string;
    allTicks: boolean;

    constructor(tableName: string, startTickColumn: string,
                foreignKeyNames: string[], otherColumnNames: string[],
                ticksPerEvent: string, parserType: ParserType, baseUrl: string,
                keyPlayerColumns: string, allTicks: string) {
        this.tableName = tableName;
        this.foreignKeyNames = foreignKeyNames;
        this.otherColumnNames = otherColumnNames;
        this.startTickColumn = parseInt(startTickColumn);
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
        this.baseUrl = baseUrl
        this.keyPlayerColumns = []
        if (keyPlayerColumns.length > 0) {
            for (const keyPlayerColumn of keyPlayerColumns.split(";")) {
                this.keyPlayerColumns.push(parseInt(keyPlayerColumn))
            }
        }
        this.allTicks = parseBool(allTicks)
    }

    parseOneLine(currentLine: string[]) {
        const id = parseInt(currentLine[0]);
        const foreignKeysStart = 1;
        const otherColumnsStart = foreignKeysStart + this.foreignKeyNames.length;
        if (this.parserType == ParserType.tick) {
            gameData.ticksTable.push(
                new TickRow(
                    id, this,
                    currentLine.slice(foreignKeysStart,
                        foreignKeysStart + this.foreignKeyNames.length)
                        .map(s => parseInt(s)),
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
                        .map(s => parseInt(s)),
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
                        .map(s => parseInt(s)),
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
                        .map(s => parseInt(s)),
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
                        .map(s => parseInt(s)),
                    currentLine.slice(otherColumnsStart, currentLine.length)
                )
            )
        }
        // clusters last for all times
        else if (this.allTicks) {
            gameData.clusters.get(this.tableName).push(
                new Row(
                    id, this,
                    currentLine.slice(foreignKeysStart,
                        foreignKeysStart + this.foreignKeyNames.length)
                        .map(s => parseInt(s)),
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
                        .map(s => parseInt(s)),
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

export class RangeIndexEntry {
    minId: number;
    maxId: number;
}
export type RangeIndex = RangeIndexEntry[];
export const tickTableName = "ticks"
export const gameTableName = "games"
export const roundTableName = "rounds"
export const playerAtTickTableName = "playerAtTick"
export const playersTableName = "players"
export const tablesNotFilteredByRound = [gameTableName, roundTableName, playersTableName]
export const tablesNotIndexedByTick = tablesNotFilteredByRound.concat([tickTableName])
export class GameData {
    tableNames: string[] = [];
    parsers: Map<string, Parser> = new Map<string, Parser>();
    gamesTable: GameRow[] = [];
    roundsTable: RoundRow[] = [];
    roundIdToIndex: Map<number, number> = new Map<number, number>();
    ticksTable: TickRow[] = [];
    playersTable: PlayerRow[] = [];
    playerIdToIndex: Map<number, number> = new Map<number, number>();
    playerAtTicksTable: PlayerAtTickRow[];
    ticksToPlayerAtTick: RangeIndex = [];
    tables: Map<string, Row[]> =
        new Map<string, Row[]>();
    ticksToOtherTablesIndices: Map<string, IntervalTree<number>> =
        new Map<string, IntervalTree<number>>();
    clusters: Map<string, Row[]> =
        new Map<string, Row[]>();

    getRound(tickData: TickRow) : RoundRow {
        if (this.roundIdToIndex.size == 0) {
            for (let i = 0; i < this.roundsTable.length; i++) {
                this.roundIdToIndex.set(this.roundsTable[i].id, i);
            }
        }
        return this.roundsTable[this.roundIdToIndex.get(tickData.roundId)]
    }

    getPlayersAtTick(tickData: TickRow) : PlayerAtTickRow[] {
        const tickIndex = tickData.id - this.ticksTable[0].id;
        if (tickIndex >= this.ticksToPlayerAtTick.length ||
            this.ticksToPlayerAtTick[tickIndex].minId == -1) {
            return [];
        }
        let result: PlayerAtTickRow[] = [];
        for (let i = this.ticksToPlayerAtTick[tickIndex].minId;
             i <= this.ticksToPlayerAtTick[tickIndex].maxId; i++) {
            result.push(this.playerAtTicksTable[i])
        }
        return result;
    }

    getPlayerByIndex(playerId: number) : PlayerRow {
        if (this.playerIdToIndex.size == 0) {
            for (let i = 0; i < this.playersTable.length; i++) {
                this.playerIdToIndex.set(this.playersTable[i].id, i);
            }
        }
        return this.playersTable[this.playerIdToIndex.get(playerId)];
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
        target.ticksToPlayerAtTick = this.ticksToPlayerAtTick;
        target.tables = this.tables;
        target.ticksToOtherTablesIndices = this.ticksToOtherTablesIndices;
        target.clusters = this.clusters;
    }

    clear() {
        this.roundIdToIndex.clear()
        this.ticksTable = [];
        this.playersTable = [];
        this.playerIdToIndex.clear();
        this.playerAtTicksTable = [];
        for (const key of Array.from(this.tables.keys())) {
            this.tables.set(key, []);
        }
        for (const key of Array.from(this.ticksToOtherTablesIndices.keys())) {
            this.ticksToOtherTablesIndices.set(key, new IntervalTree<number>())
        }
        this.ticksToPlayerAtTick = [];
        for (const key of Array.from(this.clusters.keys())) {
            this.clusters.set(key, []);
        }
    }
}

export function getTickToOtherTableIndex(gameData: GameData, tableName: string):
    IntervalTree<number> {
    if (tableName == "none") {
        return null
    }
    else if (gameData.tableNames.includes(tableName)) {
        return gameData.ticksToOtherTablesIndices.get(tableName)
    }
    else {
        throw new Error("getEventIndex for invalid tableName string " + tableName)
    }
}