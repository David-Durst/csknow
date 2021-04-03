import {PositionRow, DemoData, GameData} from "./tables";

function generatePositionsToEventsTable(position: PositionRow[],
                                               events: DemoData[],
                                               positionToEvent: Map<number, number[]>,
                                               getEventLength: (index: number) => number
                                               ) {
    for (let eventIndex = 0; eventIndex < events.length; eventIndex++) {
        let curEvent = events[eventIndex]
        let endTick = getEventLength(eventIndex);
        for (let curTick = curEvent.demoTickNumber;
             curTick <= position[position.length-1].demoTickNumber &&
                curTick < endTick;
             curTick++) {
            if (!positionToEvent.has(curTick)) {
                positionToEvent.set(curTick, [])
            }
            let values = positionToEvent.get(curTick)
            values.push(eventIndex)
            positionToEvent.set(curTick, values);
        }
    }
}

export function indexEventsForGame(gameData: GameData) {
    generatePositionsToEventsTable(gameData.position, gameData.spotted,
        gameData.positionToSpotted,
        (_: number) => gameData.spottedParser.ticksPerEvent)
    generatePositionsToEventsTable(gameData.position, gameData.weaponFire,
        gameData.positionToWeaponFire,
        (_: number) => gameData.weaponFireParser.ticksPerEvent)
    generatePositionsToEventsTable(gameData.position, gameData.playerHurt,
        gameData.positionToPlayerHurt,
        (_: number) => gameData.playerHurtParser.ticksPerEvent)
    generatePositionsToEventsTable(gameData.position, gameData.grenades,
        gameData.positionToGrenades,
        (_: number) => gameData.grenadeParser.ticksPerEvent)
    generatePositionsToEventsTable(gameData.position, gameData.kills,
        gameData.positionToKills,
        (_: number) => gameData.killsParser.ticksPerEvent)
    for (let dataName of gameData.downloadedDataNames) {
        let getTicksPerEvent = function (index: number): number {
            if (gameData.downloadedParsers.get(dataName).variableLength) {
                return parseInt(
                    gameData.downloadedData.get(dataName)[index]
                    .getColumns()[gameData.downloadedParsers.get(dataName).ticksColumn]
                )
            }
            else {
                return gameData.downloadedParsers.get(dataName).ticksPerEvent
            }
        }
        generatePositionsToEventsTable(gameData.position,
            gameData.downloadedData.get(dataName),
            gameData.downloadedPositionToEvent.get(dataName),
            getTicksPerEvent)
    }
}