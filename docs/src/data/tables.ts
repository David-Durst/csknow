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

export let rowTypes: string[] = ["ticks"]

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
        this.health = parseInt(otherColumnValues[7]);
        this.armor = parseInt(otherColumnValues[8]);
        this.isAlive = parseBool(otherColumnValues[9]);
    }
}

enum ParserType {
    tick,
    playerAtTick,
    other
}

export class Parser {
    tempLineContainer: string = "";
    tableName: string
    foreignKeyNames: string[]
    otherColumnNames: string[];
    rowType: number;
    // if variable number of ticks per event (variableLength), set ticks column
    // otherwise set ticksPerEvent
    variableLength: boolean;
    ticksPerEvent: number;
    ticksColumn: number;
    parserType: ParserType;

    constructor(tableName: string, foreignKeyNames: string[],
                otherColumnNames: string[], rowType: number,
                ticksPerEvent: string, parserType: ParserType) {
        this.tableName = tableName;
        this.foreignKeyNames = foreignKeyNames;
        this.otherColumnNames = otherColumnNames;
        this.rowType = rowType;
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
        if (this.parserType == ParserType.tick) {
            gameData.ticksTable.push(
                new TickRow(
                    id, this,
                    currentLine.slice(foreignKeysStart,
                        foreignKeysStart + this.foreignKeyNames.length)
                        .map(parseInt),
                    currentLine.slice(2, currentLine.length)
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
                    currentLine.slice(2, currentLine.length)
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
                    currentLine.slice(2, currentLine.length)
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
export const playerAtTickTableName = "playerAtTick"
export class GameData {
    tableNames: string[] = [];
    parsers: Map<string, Parser> = new Map<string, Parser>();
    ticksTable: TickRow[];
    playerAtTicksTable: PlayerAtTickRow[];
    tables: Map<string, Row[]> =
        new Map<string, Row[]>();
    ticksToOtherTablesIndices: Map<string, Index> =
        new Map<string, Map<number, number[]>>();

    clone(target: GameData) {
        target.tableNames = this.tableNames;
        target.parsers = this.parsers;
        target.ticksTable = this.ticksTable;
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