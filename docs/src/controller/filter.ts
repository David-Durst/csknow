import {gameData} from "../data/data";
import {drawTick} from "../drawing/drawing";
import {GameData, getTickToOtherTableIndex, TickRow} from "../data/tables";
import {curEvent} from "../drawing/events";
import {customFilter, setupCustomFilters} from "./ide_filters";
import {
    setCurTickIndex,
    setTickSelectorMax,
    tickLabel,
    tickSelector
} from "./selectors";

// adding some extra entries in these arrays incase extra players in server
// like casters
let lastPlayerXs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
let lastPlayerYs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
let lastPlayerZs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
function fixAfterDeath() {
    for (let t = 1; t < gameData.ticksTable.length; t++) {
        const tickData = gameData.ticksTable[t]
        const players = gameData.getPlayersAtTick(tickData)
        for (let p = 0; p < players.length; p++) {
            if (!players[p].isAlive) {
                players[p].posX = lastPlayerXs[p];
                players[p].posY = lastPlayerYs[p];
                players[p].posZ = lastPlayerZs[p];
            }
            else {
                lastPlayerXs[p] = players[p].posX
                lastPlayerYs[p] = players[p].posY
                lastPlayerZs[p] = players[p].posZ
            }
        }
    }
}

function makePitchNeg90To90() {
    for (let t = 1; t < gameData.ticksTable.length; t++) {
        const tickData = gameData.ticksTable[t]
        const players = gameData.getPlayersAtTick(tickData);
        for (let p = 0; p < players.length; p++) {
            if (players[p].viewY > 260.0) {
                players[p].viewY -= 360
            }
        }
    }
}

function alwaysFilter() {
    fixAfterDeath()
    makePitchNeg90To90()
    gameData.clone(filteredData)
}

export let filteredData: GameData = new GameData()

export function filterRegion(minX: number, minY: number, maxX: number,
                             maxY: number): boolean {
    let matchingTicks: TickRow[] = []
    for (let t = 0; t < filteredData.ticksTable.length; t++) {
        let players = filteredData.getPlayersAtTick(filteredData.ticksTable[t])
        for (let p = 0; p < 10; p++) {
            if (players[p].isAlive &&
                players[p].posX >= minX &&
                players[p].posX <= maxX &&
                players[p].posY >= minY &&
                players[p].posY <= maxY) {
                matchingTicks.push(filteredData.ticksTable[t])
                break
            }
        }
    }
    if (matchingTicks.length == 0) {
        return false;
    }
    filteredData.ticksTable = matchingTicks
    setTickSelectorMax(filteredData.ticksTable.length - 1)
    setCurTickIndex(0);
    return true;
}

let shouldFilterEvents: boolean = false
function filterEvent() {
    if (curEvent === "none" || !shouldFilterEvents) {
        return true;
    }
    let matchingPositions: TickRow[] = []
    const index = getTickToOtherTableIndex(filteredData, curEvent)
    for (let t = 0; t < filteredData.ticksTable.length; t++) {
        if (index.has(filteredData.ticksTable[t].demoTickNumber)) {
            matchingPositions.push(filteredData.ticksTable[t])
        }
    }
    if (matchingPositions.length == 0) {
        return false;
    }
    filteredData.ticksTable = matchingPositions
    setTickSelectorMax(filteredData.ticksTable.length - 1)
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
    if (filteredData.ticksTable.length === gameData.ticksTable.length) {
        return;
    }
    filteredData.ticksTable = gameData.ticksTable
    // reapply any existing event filters
    filterEvent()
    customFilter()
    setTickSelectorMax(filteredData.ticksTable.length - 1)
    setCurTickIndex(0);
}

export function setupMatchFilters() {
    gameData.clone(filteredData)
    alwaysFilter()
    setTickSelectorMax(filteredData.ticksTable.length - 1)
}

export function setupFilterHandlers() {
    document.querySelector<HTMLSelectElement>("#event_filter").addEventListener("click", filterEventButton)
    setupCustomFilters()
}