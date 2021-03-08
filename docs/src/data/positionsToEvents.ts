import {PositionRow, DemoData, GameData} from "./tables";

const ticksPerEvent = 32

function generatePositionsToEventsTable(position: PositionRow[],
                                               events: DemoData[],
                                               positionToEvent: Map<number, number[]>,
                                               ) {
    for (let spottedIndex = 0; spottedIndex < events.length; spottedIndex++) {
        let curSpotted = events[spottedIndex]
        for (let curTick = curSpotted.demoTickNumber;
             curTick < position.length &&
                curTick < curSpotted.demoTickNumber + ticksPerEvent;
             curTick++) {
            if (!positionToEvent.has(curTick)) {
                positionToEvent.set(curTick, [])
            }
            let values = positionToEvent.get(curTick)
            values.push(spottedIndex)
            positionToEvent.set(curTick, values);
        }
    }
}

export function indexEventsForGame(gameData: GameData) {
    generatePositionsToEventsTable(gameData.position, gameData.spotted,
        gameData.positionToSpotted)
    generatePositionsToEventsTable(gameData.position, gameData.weaponFire,
        gameData.positionToWeaponFire)
    generatePositionsToEventsTable(gameData.position, gameData.playerHurt,
        gameData.positionToPlayerHurt)
    generatePositionsToEventsTable(gameData.position, gameData.grenades,
        gameData.positionToGrenades)
    generatePositionsToEventsTable(gameData.position, gameData.kills,
        gameData.positionToKills)
}