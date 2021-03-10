import {gameData} from "../data/data";
import {drawTick} from "../drawing/drawing";
import {GameData, getEventIndex, PositionRow} from "../data/tables";
import {curEvent} from "../drawing/events";
import {customFilter, setupCustomFilters} from "./ide_filters";
import {
    setCurTickIndex,
    setTickSelectorMax, setupTickSelector,
    tickLabel,
    tickSelector
} from "./tickSelector";

let lastPlayerXs = [0,0,0,0,0,0,0,0,0,0]
let lastPlayerYs = [0,0,0,0,0,0,0,0,0,0]
let lastPlayerZs = [0,0,0,0,0,0,0,0,0,0]
function fixAfterDeath() {
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

function makePitchNeg90To90() {
    for (let t = 1; t < gameData.position.length; t++) {
        const curTick = gameData.position[t]
        for (let p = 0; p < 10; p++) {
            if (curTick.players[p].yViewDirection > 260.0) {
                curTick.players[p].yViewDirection -= 360
            }
        }
    }
}

function alwaysFilter() {
    fixAfterDeath()
    makePitchNeg90To90()
    filteredData.position = gameData.position
    filteredData.spotted = gameData.spotted
    filteredData.weaponFire = gameData.weaponFire
    filteredData.playerHurt = filteredData.playerHurt
    filteredData.grenades = filteredData.grenades
    filteredData.kills = gameData.kills
}

export let filteredData: GameData = new GameData()

export function filterRegion(minX: number, minY: number, maxX: number,
                             maxY: number): boolean {
    let matchingPositions: PositionRow[] = []
    for (let t = 0; t < filteredData.position.length; t++) {
        for (let p = 0; p < 10; p++) {
            if (filteredData.position[t].players[p].isAlive &&
                filteredData.position[t].players[p].xPosition >= minX &&
                filteredData.position[t].players[p].xPosition <= maxX &&
                filteredData.position[t].players[p].yPosition >= minY &&
                filteredData.position[t].players[p].yPosition <= maxY) {
                matchingPositions.push(filteredData.position[t])
                break
            }
        }
    }
    if (matchingPositions.length == 0) {
        return false;
    }
    filteredData.position = matchingPositions
    setTickSelectorMax(filteredData.position.length - 1)
    setCurTickIndex(0);
    return true;
}

let shouldFilterEvents: boolean = false
function filterEvent() {
    if (curEvent === "none" || !shouldFilterEvents) {
        return true;
    }
    let matchingPositions: PositionRow[] = []
    const index = getEventIndex(filteredData, curEvent)
    for (let t = 0; t < filteredData.position.length; t++) {
        if (index.has(filteredData.position[t].demoTickNumber)) {
            matchingPositions.push(filteredData.position[t])
        }
    }
    if (matchingPositions.length == 0) {
        return false;
    }
    filteredData.position = matchingPositions
    setTickSelectorMax(filteredData.position.length - 1)
    setCurTickIndex(0);
    drawTick(null)
    return true;
}

function filterEventButton() {
    shouldFilterEvents = true
    filterEvent()
}

export function stopFilteringEvents() {
    shouldFilterEvents = false
}

export function clearFilterData() {
    if (filteredData.position.length === gameData.position.length) {
        return;
    }
    filteredData.position = gameData.position
    // reapply any existing event filters
    filterEvent()
    customFilter()
    setTickSelectorMax(filteredData.position.length - 1)
    setCurTickIndex(0);
}

export function setupMatchFilters() {
    gameData.clone(filteredData)
    alwaysFilter()
    setTickSelectorMax(filteredData.position.length - 1)
}

export function setupInitFilters() {
    setupTickSelector()
}

export function setupFilterHandlers() {
    document.querySelector<HTMLSelectElement>("#event_filter").addEventListener("click", filterEventButton)
    setupCustomFilters()
}