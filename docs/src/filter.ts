import {gameData} from "./data";

let lastPlayerXs = [0,0,0,0,0,0,0,0,0,0]
let lastPlayerYs = [0,0,0,0,0,0,0,0,0,0]
let lastPlayerZs = [0,0,0,0,0,0,0,0,0,0]
export function alwaysFilter() {
    console.log("applying always filter")
    for (let t = 1; t < gameData.position.length; t++) {
        const curTick = gameData.position[t]
        for (let p = 0; p < 10; p++) {
            if (!curTick.players[p].isAlive) {
                curTick.players[p].xPosition = lastPlayerXs[p];
                curTick.players[p].yPosition = lastPlayerYs[p];
                curTick.players[p].zPosition = lastPlayerZs[p];
            }
            else {
                lastPlayerXs[p] = curTick.players[p].xPosition
                lastPlayerYs[p] = curTick.players[p].yPosition
                lastPlayerZs[p] = curTick.players[p].zPosition
            }
        }

    }
}
