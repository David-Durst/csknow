import {
    Row,
    TickRow,
    GameData,
    playerAtTickTableName,
    tickTableName,
    RangeIndex,
    RangeIndexEntry, tablesNotFilteredByRound,
} from "./tables";
import {gameData} from "./data";
import IntervalTree from "@flatten-js/interval-tree";

function generateRangeIndex(ticks: TickRow[], otherTable: Row[],
                            index: RangeIndex) {
    for (let tickIndex = 0, otherIndex = 0; tickIndex < ticks.length; tickIndex++) {
        if (otherIndex >= otherTable.length ||
            otherTable[otherIndex].getStartTick() > ticks[tickIndex].id) {
            index.set(ticks[tickIndex].id, new RangeIndexEntry())
            index.get(ticks[tickIndex].id).minId = -1;
            index.get(ticks[tickIndex].id).maxId = -1;
        }
        else {
            // sometimes have mistakes where point to 0 as uninitalized, skip entries
            for(; otherTable[otherIndex].getStartTick() <= 0 &&
                  otherTable[otherIndex].getStartTick() < ticks[tickIndex].id;
                  otherIndex++);

            index.set(ticks[tickIndex].id, new RangeIndexEntry())
            index.get(ticks[tickIndex].id).minId = otherIndex;
            for (; otherIndex < otherTable.length &&
                   otherTable[otherIndex].getStartTick() == ticks[tickIndex].id;
                   otherIndex++) ;
            index.get(ticks[tickIndex].id).maxId = otherIndex - 1;
        }
    }

}

function generateTreeIndex(otherTable: Row[], index: IntervalTree<number>,
                              getEventLength: (index: number, tick:number) => number) {
    for (let otherIndex = 0; otherIndex < otherTable.length; otherIndex++) {
        let curEvent = otherTable[otherIndex]
        let endTick = curEvent.getStartTick() + getEventLength(otherIndex, curEvent.id);
        index.insert([curEvent.getStartTick(), endTick], otherIndex)
    }
}

function generatePerTickAimIndex(eventTable: Row[], perTickAimTable: Row[], eventIdColIndex: number,
                                 index: Map<number, number[]>) {
    // first generate array for every event
    for (let eventIndex = 0; eventIndex < eventTable.length; eventIndex++) {
        index.set(eventIndex, [])
    }
    for (let perTickAimIndex = 0; perTickAimIndex < perTickAimTable.length; perTickAimIndex++) {
        const aimPerEvent = index.get(perTickAimTable[perTickAimIndex].foreignKeyValues[eventIdColIndex])
        aimPerEvent.push(perTickAimIndex)
    }
}

export function indexEventsForRound(gameData: GameData) {
    gameData.ticksToPlayerAtTick.clear()
    generateRangeIndex(gameData.ticksTable, gameData.playerAtTicksTable,
        gameData.ticksToPlayerAtTick)

    for (let dataName of gameData.tableNames) {
        const curParser = gameData.parsers.get(dataName)
        // skip if non-temporal, longer time scale than a round, tick data, or already indexed by ticks to PAT
        if (tablesNotFilteredByRound.includes(dataName)
            || dataName == tickTableName || dataName == playerAtTickTableName
            || curParser.nonTemporal) {
            continue;
        }
        let getTicksPerEvent = function (index: number, tick: number): number {
            if (gameData.parsers.get(dataName).variableLength) {
                return gameData.tables.get(dataName)[index]
                    .foreignKeyValues[curParser.ticksColumn]
            }
            else {
                return curParser.ticksPerEvent
            }
        }
        gameData.ticksToOtherTablesIndices.set(dataName, new IntervalTree<number>());
        generateTreeIndex(gameData.tables.get(dataName),
            gameData.ticksToOtherTablesIndices.get(dataName),
            getTicksPerEvent)
        // if have a per tick aim table, create index for that
        if (curParser.havePerTickAimTable) {
            gameData.eventToPerTickAimTablesIndices.set(dataName, new Map<number, number[]>());
            generatePerTickAimIndex(gameData.tables.get(dataName), gameData.tables.get(curParser.perTickAimTable),
                gameData.parsers.get(curParser.perTickAimTable).eventIdColumn,
                gameData.eventToPerTickAimTablesIndices.get(dataName))
        }
    }
}