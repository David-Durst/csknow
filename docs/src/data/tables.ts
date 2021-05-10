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

    constructor(tableName: string, foreignKeyNames: string[],
                otherColumnNames: string[], rowType: number,
                ticksPerEvent: string) {
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
    }

    parseOneLine(currentLine: string[]): any {
        const id = parseInt(currentLine[0]);
        const foreignKeysStart = 1;
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

    reader: any = null
    setReader(readerInput: any) {
        this.reader = readerInput
    }

}

export class GameData {
    tableNames: string[] = [];
    parsers: Map<string, Parser> = new Map<string, Parser>();
    tables: Map<string, Row[]> =
        new Map<string, Row[]>();
    ticksToOtherTablesIndices: Map<string, Map<number, number[]>> =
        new Map<string, Map<number, number[]>>();

    clone(target: GameData) {
        target.tableNames = this.tableNames;
        target.parsers = this.parsers
        target.tables = this.tables
        target.ticksToOtherTablesIndices = this.ticksToOtherTablesIndices
    }
}

export function getTickToOtherTableIndex(gameData: GameData, event: string):
    Map<number, number[]> {
    if (event == "none") {
        return null
    }
    else if (gameData.tableNames.includes(event)) {
        return gameData.ticksToOtherTablesIndices.get(event)
    }
    else {
        throw new Error("getEventIndex for invalid event string " + event)
    }
}