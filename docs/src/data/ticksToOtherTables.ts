import {
    Row,
    TickRow,
    GameData,
    HashmapIndex,
    playerAtTickTableName,
    tablesNotIndexedByTick,
    tickTableName,
    RangeIndex,
    RangeIndexEntry,
} from "./tables";
import {gameData} from "./data";

function generateRangeIndex(ticks: TickRow[], otherTable: Row[],
                            index: RangeIndex) {
    for (let tickIndex = 0, otherIndex = 0; tickIndex < ticks.length; tickIndex++) {
        if (otherIndex >= otherTable.length ||
            otherTable[otherIndex].getStartTick() > ticks[tickIndex].id) {
            index.push(new RangeIndexEntry())
            index[tickIndex].minId = -1;
            index[tickIndex].maxId = -1;
        }
        else {
            // sometimes have mistakes where point to 0 as uninitalized, skip entries
            for(; otherTable[otherIndex].getStartTick() <= 0 &&
                  otherTable[otherIndex].getStartTick() < ticks[tickIndex].id;
                  otherIndex++);

            index.push(new RangeIndexEntry())
            index[tickIndex].minId = otherIndex;
            for (; otherIndex < otherTable.length &&
                   otherTable[otherIndex].getStartTick() == ticks[tickIndex].id;
                   otherIndex++) ;
            index[tickIndex].maxId = otherIndex - 1;
        }
    }

}

function generateHashmapIndex(ticks: TickRow[], otherTable: Row[],
                              index: HashmapIndex,
                              getEventLength: (index: number, tick:number) => number) {
    for (let otherIndex = 0; otherIndex < otherTable.length; otherIndex++) {
        let curEvent = otherTable[otherIndex]
        let endTick = curEvent.getStartTick() + getEventLength(otherIndex, curEvent.id);
        for (let curTick = curEvent.getStartTick();
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

export function indexEventsForRound(gameData: GameData) {
    generateRangeIndex(gameData.ticksTable, gameData.playerAtTicksTable,
        gameData.ticksToPlayerAtTick)

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
        generateHashmapIndex(gameData.ticksTable,
            gameData.tables.get(dataName),
            gameData.ticksToOtherTablesIndices.get(dataName),
            getTicksPerEvent)
    }
}