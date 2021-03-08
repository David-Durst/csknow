import {PositionRow, GameData} from "./tables";

const ticksPerEvent = 32

export function generatePositionsToEventsTable(position: PositionRow[],
                                               events:
                                               positionToEvent: Map<number, number[]>,
                                               ) {
    for (let spottedIndex = 0; spottedIndex < gameData.spotted.length; spottedIndex++) {
        let curSpotted = gameData["spotted"][spottedIndex]
        for (let curTick = curSpotted.tickNumber;
             curTick < gameData["position"].length &&
                curTick < curSpotted.tickNumber + ticksPerEvent;
             curTick++) {
            if (!gameData["positionToSpotted"].has(curTick)) {
                gameData["positionToSpotted"].set(curTick, [])
            }
            let values = gameData["positionToSpotted"].get(curTick)
            values.push(spottedIndex)
            gameData.positionToSpotted.set(curTick, values);
        }
    }
    let curWeaponFire = 0
    let curPlayerHurt = 0
    let curGrenade = 0
    let curKill = 0
    for (let curTick = 0; curTick < gameData.position.length; curTick++) {
        for (let backTicks = 0; backTicks > ticksKeepEvent &&
                curTick - backTicks >= 0; backTicks--) {
            const prevTick = curTick - backTicks;

        }
    }
}