import {Row, GameData, rowTypes} from "./tables";

function generateTickToOtherTableIndex(ticks: Row[], otherTable: Row[],
                                       index: Map<number, number[]>,
                                       getEventLength: (index: number, tick:number) => number
                                               ) {
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
    for (let dataName of gameData.tableNames) {
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
        generateTickToOtherTableIndex(gameData.tables.get(rowTypes[0]),
            gameData.tables.get(dataName),
            gameData.ticksToOtherTablesIndices.get(dataName),
            getTicksPerEvent)
    }
}