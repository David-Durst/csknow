import {GameData, gameData, PositionRow} from "./data";
import {match} from "assert";
import {drawTick} from "./drawing";

export let tickSelector: HTMLInputElement = null;
export let tickLabel: HTMLLabelElement = null;
function setTickSelectorMax(value: number) {
    tickSelector.max = value.toString()
}
export function getCurTickIndex(): number {
    return parseInt(tickSelector.value)
}
function setCurTickIndex(value: number) {
    tickSelector.value = value.toString()
}
export function setTickLabel(value: number) {
    tickLabel.innerHTML = value.toString()
}

let lastPlayerXs = [0,0,0,0,0,0,0,0,0,0]
let lastPlayerYs = [0,0,0,0,0,0,0,0,0,0]
let lastPlayerZs = [0,0,0,0,0,0,0,0,0,0]
function alwaysFilter() {
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

// track where this tick so can keep slider pointing at same tick
// when adding/removing filters
let tickIndexFilteredToOrig: number[] = []
export function filterRegion(minX: number, minY: number, maxX: number,
                             maxY: number): boolean {
    let curTickPrefiltering = getCurTickIndex()
    let curTickPostFiltering = -1;
    let matchingPositions: PositionRow[] = []
    tickIndexFilteredToOrig = []
    for (let t = 0; t < gameData.position.length; t++) {
        for (let p = 0; p < 10; p++) {
            if (gameData.position[t].players[p].isAlive &&
                gameData.position[t].players[p].xPosition >= minX &&
                gameData.position[t].players[p].xPosition <= maxX &&
                gameData.position[t].players[p].yPosition >= minY &&
                gameData.position[t].players[p].yPosition <= maxY) {
                matchingPositions.push(gameData.position[t])
                tickIndexFilteredToOrig.push(t)
                if (t == curTickPrefiltering) {
                    curTickPostFiltering = matchingPositions.length - 1
                }
                break
            }
        }
    }
    if (matchingPositions.length == 0) {
        return false;
    }
    filteredData.position = matchingPositions
    setTickSelectorMax(filteredData.position.length - 1)
    setCurTickIndex(curTickPostFiltering);
    return true;
}

export function clearRegionFilterData() {
    if (filteredData.position.length === gameData.position.length) {
        return;
    }
    filteredData.position = gameData.position
    setTickSelectorMax(filteredData.position.length - 1)
    setCurTickIndex(tickIndexFilteredToOrig[getCurTickIndex()]);
}

export function setupMatchFilters() {
    alwaysFilter()
    setTickSelectorMax(filteredData.position.length - 1)
}

export function setupInitFilters() {
    tickSelector = document.querySelector<HTMLInputElement>("#tick-selector")
    tickLabel = document.querySelector<HTMLLabelElement>("#cur-tick")
    tickLabel.innerHTML = "0"
}
