import {PositionRow, DemoData, GameData} from "./tables";

function generatePositionsToEventsTable(position: PositionRow[],
                                               events: DemoData[],
                                               positionToEvent: Map<number, number[]>,
                                               ticksPerEvent: number
                                               ) {
    for (let eventIndex = 0; eventIndex < events.length; eventIndex++) {
        let curEvent = events[eventIndex]
        for (let curTick = curEvent.demoTickNumber;
             curTick <= position[position.length-1].demoTickNumber &&
                curTick < curEvent.demoTickNumber + ticksPerEvent;
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
        gameData.positionToSpotted, gameData.spottedParser.ticksPerEvent)
    generatePositionsToEventsTable(gameData.position, gameData.weaponFire,
        gameData.positionToWeaponFire, gameData.weaponFireParser.ticksPerEvent)
    generatePositionsToEventsTable(gameData.position, gameData.playerHurt,
        gameData.positionToPlayerHurt, gameData.playerHurtParser.ticksPerEvent)
    generatePositionsToEventsTable(gameData.position, gameData.grenades,
        gameData.positionToGrenades, gameData.grenadeParser.ticksPerEvent)
    generatePositionsToEventsTable(gameData.position, gameData.kills,
        gameData.positionToKills, gameData.killsParser.ticksPerEvent)
    for (let dataName of gameData.downloadedDataNames) {
        generatePositionsToEventsTable(gameData.position,
            gameData.downloadedData.get(dataName),
            gameData.downloadedPositionToEvent.get(dataName),
            gameData.downloadedParsers.get(dataName).ticksPerEvent)
    }
}