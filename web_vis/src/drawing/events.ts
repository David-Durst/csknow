import {
    GameData,
    getTickToOtherTableIndex,
    PlayerAtTickRow,
    playerAtTickTableName, PlayerRow,
    Row,
    TickRow,
} from "../data/tables";
import IntervalTree from "@flatten-js/interval-tree";
import {gameData, INVALID_ID} from "../data/data";
import {drawTick} from "../drawing/drawing";

let eventSelector: HTMLSelectElement = null
let eventDiv: HTMLDivElement = null
let eventIdSelector: HTMLSelectElement = null
let eventIdLabel: HTMLSpanElement = null
let overlaySelector: HTMLSelectElement = null
export let curEvent: string = "none"
export let selectedEventId: number = INVALID_ID
export let activeEvent: Row = null
export let curOverlay: string = "none"
export let displayMouseData = true
let mouseDataDisplayButton: HTMLButtonElement = null

export const DEFAULT_ALIVE_STRING = "o"
export const DEFAULT_DEAD_STRING = "x"
function basicPlayerText(gameData: GameData, tickData: TickRow,
                         playerIndex: number): string {
    if (gameData.getPlayersAtTick(tickData)[playerIndex].isAlive) {
        return DEFAULT_ALIVE_STRING
    }
    else {
        return DEFAULT_DEAD_STRING
    }
}

export function getPlayersText(tickData: TickRow, gameData: GameData): Map<number, string> {
    // would be great if could draw more than 1 player
    // but 2 players can shoot each other same tick and not able to visualize that right now
    let result: Map<number, string> = new Map()
    const index = getTickToOtherTableIndex(gameData, curEvent)
    const players: PlayerAtTickRow[] = gameData.getPlayersAtTick(tickData)
    // if no event, do nothing special
    if (curEvent == "none" || !index.intersect_any([tickData.id, tickData.id]) ||
        !gameData.parsers.get(curEvent).havePlayerLabels) {
        for (let p = 0; p < players.length; p++) {
            result.set(players[p].playerId, basicPlayerText(gameData, tickData, p))
        }
        activeEvent = null
        return result
    }

    // if event, get min key player number for each player
    const eventArray = gameData.tables.get(curEvent)
    const eventsForTick = index.search([tickData.id, tickData.id])
    // see updateEventId for how curEventId is set
    if (selectedEventId == INVALID_ID) {
        activeEvent = eventArray[Math.min(...eventsForTick)]
    }
    else {
        for (let i = 0; i < eventArray.length; i++) {
            if (selectedEventId == eventArray[i].id) {
                activeEvent = eventArray[i]
            }
        }
    }
    const parser = gameData.parsers.get(curEvent)
    const playersToLabel = activeEvent.otherColumnValues[parser.playersToLabelColumn].split(";").map(x => parseInt(x))
    const playerLabelIndices = activeEvent.otherColumnValues[parser.playerLabelIndicesColumn].split(";").map(x => parseInt(x))
    for (let p = 0; p < players.length; p++) {
        const indexOfPlayerInLabelsArray = playersToLabel.indexOf(players[p].playerId)
        if (players[p].isAlive && indexOfPlayerInLabelsArray != -1) {
            result.set(players[p].playerId, parser.playerLabels[playerLabelIndices[indexOfPlayerInLabelsArray]])
        }
        else {
            result.set(players[p].playerId, basicPlayerText(gameData, tickData, p))
        }
    }
    /*

    const playerIDToLocalPATIndex: Map<number, number> = new Map<number, number>()
    let minKeyPlayerNumbers: number[] = [];
    const bogusMinKeyPlayerNumber = 100
    for (let p = 0; p < players.length; p++) {
        minKeyPlayerNumbers[p] = bogusMinKeyPlayerNumber
    }
    for (let eIndex = 0; eIndex < eventsForTick.length; eIndex++)  {
        let event = eventArray[eventsForTick[eIndex]]
        for (const keyPlayerIndex of eventArray[0].parser.keyPlayerColumns) {
            const localID =
                playerIDToLocalPATIndex.get(event.foreignKeyValues[keyPlayerIndex])
            if (!players[localID].isAlive) {
                continue;
            }
            minKeyPlayerNumbers[localID] =
                Math.min(minKeyPlayerNumbers[localID], keyPlayerIndex);
        }
    }
    for (let p = 0; p < players.length; p++) {
        if (minKeyPlayerNumbers[p] < bogusMinKeyPlayerNumber) {
            result[p] = minKeyPlayerNumbers[p].toString()
        }
    }

     */
    return result
}

export function setEventText(tickData: TickRow, gameData: GameData) {
    if (curEvent == "none") {
        return
    }
    eventDiv.innerHTML = ""
    if (curEvent == playerAtTickTableName) {
        const patRows = gameData.getPlayersAtTick(tickData)
        for (let patRowIndex = 0; patRowIndex < patRows.length; patRowIndex++) {
            eventDiv.innerHTML += patRows[patRowIndex].getHTML()
        }
    }
    else {
        const index = getTickToOtherTableIndex(gameData, curEvent)
        if (index.intersect_any([tickData.id, tickData.id])) {
            const events = index.search([tickData.id, tickData.id])
            const eventArray = gameData.tables.get(curEvent)
            for (let eIndex = events.length - 1; eIndex >= 0; eIndex--) {
                eventDiv.innerHTML += eventArray[events[eIndex]].getHTML()
            }
        }
    }
}

function toggleMouseDisplay() {
    if (displayMouseData) {
        displayMouseData = false
        mouseDataDisplayButton.innerText = "show aim data"
    }
    else {
        displayMouseData = true
        mouseDataDisplayButton.innerText = "hide aim data"
    }
    drawTick(null)
}

export function updateEventIdAndSelector(tickData: TickRow) {
    const parser = gameData.parsers.get(curEvent)
    if (curEvent == "none" || !parser.havePlayerLabels) {
        eventIdLabel.style.display = "none"
        eventIdSelector.style.display = "none"
        eventIdSelector.options.length = 0
        eventIdSelector.options.add(new Option("default", "-1"))
        mouseDataDisplayButton.style.display = "none"
    }
    else {
        eventIdLabel.style.display = "inline"
        eventIdSelector.style.display = "inline-block"
        eventIdSelector.options.length = 0
        eventIdSelector.options.add(new Option("default", "-1"))
        const index = getTickToOtherTableIndex(gameData, curEvent)
        const eventArray = gameData.tables.get(curEvent)
        const eventsForTick = index.search([tickData.id, tickData.id])
        // default to non-valid eventId index
        let eventIdIndex = 0
        for (let i = 0; i < eventsForTick.length; i++) {
            const curEventRow = eventArray[eventsForTick[i]]
            if (curEventRow.id == selectedEventId) {
                // plus 1 as need to account for default value
                eventIdIndex = i + 1
            }
            let eventName = curEventRow.otherColumnValues[parser.playersToLabelColumn]
            eventIdSelector.options.add(new Option(eventName, curEventRow.id.toString()))
        }
        eventIdSelector.selectedIndex = eventIdIndex
        // if old selected event id no longer value, just go with default setting
        if (eventIdIndex == 0) {
            selectedEventId = INVALID_ID
        }
        if (parser.havePerTickAimTable) {
            mouseDataDisplayButton.style.display = "inline-block"
        }
    }
}

export function setEventsOverlaysToDraw() {
    curEvent = eventSelector.value
    selectedEventId = parseInt(eventIdSelector.value)
    curOverlay = overlaySelector.value;
}

export function setupEventDrawing() {
    eventSelector = document.querySelector<HTMLSelectElement>("#event-type")
    curEvent = eventSelector.value;
    eventDiv = document.querySelector<HTMLDivElement>("#events")
    eventIdSelector = document.querySelector<HTMLSelectElement>("#event-id-selector")
    eventIdLabel = document.querySelector<HTMLSelectElement>("#event-id-label")
    overlaySelector = document.querySelector<HTMLSelectElement>("#overlay-type")
    curOverlay = overlaySelector.value;
    mouseDataDisplayButton = document.querySelector<HTMLButtonElement>("#event-mouse-display")
    mouseDataDisplayButton.addEventListener("click", toggleMouseDisplay)
}