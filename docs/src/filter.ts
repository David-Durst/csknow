import {GameData, gameData, PositionRow} from "./data";
import {match} from "assert";

let lastPlayerXs = [0,0,0,0,0,0,0,0,0,0]
let lastPlayerYs = [0,0,0,0,0,0,0,0,0,0]
let lastPlayerZs = [0,0,0,0,0,0,0,0,0,0]
export function alwaysFilter() {
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
    filteredData.position = gameData.position
    filteredData.spotted = gameData.spotted
    filteredData.weaponFire = gameData.weaponFire
    filteredData.playerHurt = filteredData.playerHurt
    filteredData.grenades = filteredData.grenades
    filteredData.kills = gameData.kills
}

export let filteredData: GameData = new GameData()

export function filterRegion(minX: number, maxX: number, minY: number,
                             maxY: number) {
    let matchingPositions: PositionRow[] = []
    for (let t = 0; t < gameData.position.length; t++) {
        for (let p = 0; p < 10; p++) {
            if (gameData.position[t].players[p].isAlive &&
                gameData.position[t].players[p].xPosition >= minX &&
                gameData.position[t].players[p].xPosition >= maxX &&
                gameData.position[t].players[p].yPosition <= minY &&
                gameData.position[t].players[p].yPosition >= maxY) {
                matchingPositions.push(gameData.position[t])
                continue
            }
        }
    }
    filteredData.position = matchingPositions
}

export function clearFilterRegion() {
    filteredData.position = gameData.position
}


