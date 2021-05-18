import {
    Row,
    TickRow,
    GameData,
    Index,
    playerAtTickTableName, tablesNotIndexedByTick, tickTableName,
} from "./tables";
import {gameData} from "./data";

function generateTicksToOtherTableIndex(ticks: TickRow[], otherTable: Row[],
                                       index: Index,
                                       getEventLength: (index: number, tick:number)
                                           => number) {
    for (let otherIndex = 0; otherIndex < otherTable.length; otherIndex++) {
        let curEvent = otherTable[otherIndex]
        let endTick = getEventLength(otherIndex, curEvent.id);
        for (let curTick = curEvent.id;
             curTick <= ticks[ticks.length-1].id &&
                curTick < endTick;
             curTick++) {
            if (!index.has(curTick)) {
                index.set(curTick, [])
            }
            let values = index.get(curTick)
            values.push(otherIndex)
            index.set(curTick, values);
        }
    }
}

export function indexEventsForGame(gameData: GameData) {
    generateTicksToOtherTableIndex(gameData.ticksTable, gameData.playerAtTicksTable,
        gameData.ticksToOtherTablesIndices.get(playerAtTickTableName),
        (_: number, tick: number) => tick +
            gameData.parsers.get(playerAtTickTableName).ticksPerEvent)

    for (let dataName of gameData.tableNames) {
        if (tablesNotIndexedByTick.includes(dataName)
            || dataName == playerAtTickTableName || dataName == tickTableName) {
            continue;
        }
        let getTicksPerEvent = function (index: number, tick: number): number {
            if (gameData.parsers.get(dataName).variableLength) {
                return parseInt(
                    gameData.tables.get(dataName)[index]
                    .getColumns()[gameData.parsers.get(dataName).ticksColumn]
                )
            }
            else {
                return tick + gameData.parsers.get(dataName).ticksPerEvent
            }
        }
        generateTicksToOtherTableIndex(gameData.ticksTable,
            gameData.tables.get(dataName),
            gameData.ticksToOtherTablesIndices.get(dataName),
            getTicksPerEvent)
    }
}