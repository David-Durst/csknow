import {PositionRow, DemoData, GameData} from "./tables";

const ticksPerEvent = 32

function generatePositionsToEventsTable(position: PositionRow[],
                                               events: DemoData[],
                                               positionToEvent: Map<number, number[]>,
                                               ) {
    for (let eventIndex = 0; eventIndex < events.length; eventIndex++) {
        let curEvent = events[eventIndex]
        for (let curTick = curEvent.demoTickNumber;
             curTick < position.length &&
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
        gameData.positionToSpotted)
    generatePositionsToEventsTable(gameData.position, gameData.weaponFire,
        gameData.positionToWeaponFire)
    generatePositionsToEventsTable(gameData.position, gameData.playerHurt,
        gameData.positionToPlayerHurt)
    generatePositionsToEventsTable(gameData.position, gameData.grenades,
        gameData.positionToGrenades)
    generatePositionsToEventsTable(gameData.position, gameData.kills,
        gameData.positionToKills)
    generatePositionsToEventsTable(gameData.position, gameData.wallers,
        gameData.positionToWallers)
}