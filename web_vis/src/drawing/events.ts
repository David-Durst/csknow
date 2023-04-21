import {
    GameData,
    getTickToOtherTableIndex,
    PlayerAtTickRow,
    playerAtTickTableName, PlayerRow,
    Row,
    TickRow, unPackVec3ListStr, Vec3,
} from "../data/tables";
import IntervalTree from "@flatten-js/interval-tree";
import {gameData, INVALID_ID} from "../data/data";
import {drawTick} from "../drawing/drawing";
import {start} from "repl";

let eventSelector: HTMLSelectElement = null
let eventDiv: HTMLDivElement = null
let eventIdSelector: HTMLSelectElement = null
let eventIdLabel: HTMLSpanElement = null
let overlaySelector: HTMLSelectElement = null
export let curEvent: string = "none"
export let selectedEventId: number = INVALID_ID
let priorEventId: number = INVALID_ID
export let activeEvent: Row = null
export let curOverlay: string = "none"
export let displayMouseData = true
export let zoomMouseData = true
export let showDistribution = false
let mouseDataDisplayButton: HTMLButtonElement = null
let mouseDataZoomButton: HTMLButtonElement = null
let showDistributionButton: HTMLButtonElement = null

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
        !gameData.parsers.get(curEvent).havePlayerLabels || selectedEventId == -2) {
        for (let p = 0; p < players.length; p++) {
            result.set(players[p].playerId, basicPlayerText(gameData, tickData, p))
        }
        activeEvent = null
        return result
    }

    // if event, get min key player number for each player
    const eventArray = gameData.tables.get(curEvent)
    const eventsForTick = index.search([tickData.id, tickData.id])

    // default value for active event
    if (selectedEventId == INVALID_ID) {
        activeEvent = eventArray[Math.min(...eventsForTick)]
    }
    // use selected to override if that set, or prior is that is set as backup
    // see updateEventId for how selectEventId is set
    if (selectedEventId != INVALID_ID) {
        for (let i = 0; i < eventsForTick.length; i++) {
            const eventIndex = eventsForTick[i]
            if (selectedEventId == eventArray[eventIndex].id) {
                activeEvent = eventArray[eventIndex]
            }
        }
    }
    else if (priorEventId != INVALID_ID) {
        for (let i = 0; i < eventsForTick.length; i++) {
            const eventIndex = eventsForTick[i]
            if (priorEventId == eventArray[eventIndex].id) {
                activeEvent = eventArray[eventIndex]
            }
        }
    }
    priorEventId = activeEvent.id;

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
    if (parser.perTickPlayerLabelsQuery != "" && playersToLabel.length > 0 && showDistribution) {
        const sourcePlayerId = playersToLabel[0]
        let sourcePATId = INVALID_ID
        for (let i = gameData.ticksToPlayerAtTick.get(tickData.id).minId;
             i <= gameData.ticksToPlayerAtTick.get(tickData.id).maxId; i++) {
            if (gameData.playerAtTicksTable[i].playerId == sourcePlayerId) {
                sourcePATId = i;
                break
            }
        }
        if (sourcePATId != INVALID_ID) {
            const labelData = gameData.tables.get(parser.perTickPlayerLabelsQuery)
            const playerAndProbs = labelData[sourcePATId].otherColumnValues[0].split(";")
            for (let i = 0; i < playerAndProbs.length; i++) {
                const playerAndProb = playerAndProbs[i].split("=")
                if (parseInt(playerAndProb[0]) != INVALID_ID) {
                    result.set(parseInt(playerAndProb[0]), playerAndProb[1])
                }
            }
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

export class PosTextPosition {
    constructor(pos: Vec3, text: string) {
        this.pos = pos
        this.text = text
    }

    pos: Vec3;
    text: string;
}

export function getPosTextPositions(tickData: TickRow, gameData: GameData): Array<PosTextPosition> {
    let result = new Array<PosTextPosition>()
    const index = getTickToOtherTableIndex(gameData, curEvent)
    if (curEvent == "none" || !index.intersect_any([tickData.id, tickData.id]) ||
        !gameData.parsers.get(curEvent).havePlayerLabels || selectedEventId == -2) {
        return result
    }
    const parser = gameData.parsers.get(curEvent)
    const playersToLabel = activeEvent.otherColumnValues[parser.playersToLabelColumn].split(";").map(x => parseInt(x))
    if (parser.perTickPosLabelsQuery != "" && showDistribution) {
        const posLabelsParser = gameData.parsers.get(parser.perTickPosLabelsQuery)
        const sourcePlayerId = playersToLabel[0]
        let sourcePATId = INVALID_ID
        const labelData = gameData.tables.get(parser.perTickPosLabelsQuery)
        for (let i = gameData.ticksToPlayerAtTick.get(tickData.id).minId;
             i <= gameData.ticksToPlayerAtTick.get(tickData.id).maxId; i++) {
            if (gameData.playerAtTicksTable[i].playerId == sourcePlayerId) {
                sourcePATId = i;
                break
            }
        }
        if (sourcePATId != INVALID_ID) {
            const posProbs = labelData[sourcePATId].otherColumnValues[0].split(";")
            if (parser.havePerTickPos) {
                const aabbStr = labelData[sourcePATId].otherColumnValues[posLabelsParser.perTickPosAABBColumn]
                const aabb = unPackVec3ListStr(aabbStr)
                const labelsPerRow = Math.ceil(Math.sqrt(posProbs.length))
                const startVec3 = aabb[0]
                const endVec3 = aabb[0]
                const deltaX = (endVec3.posX - startVec3.posX) / labelsPerRow
                const deltaY = (endVec3.posY - startVec3.posY) / labelsPerRow
                const avgZ = (endVec3.posZ + startVec3.posZ) / 2
                for (let i = 0; i < posLabelsParser.posLabelPositions.length; i++) {
                    const xVal = i % labelsPerRow
                    const yVal = Math.floor(i / labelsPerRow)
                    result.push(new PosTextPosition(new Vec3(
                        startVec3.posX + deltaX * xVal,
                        startVec3.posY + deltaY * yVal,
                        avgZ), posProbs[i]))
                }
            }
            else {
                for (let i = 0; i < posLabelsParser.posLabelPositions.length; i++) {
                    result.push(new PosTextPosition(posLabelsParser.posLabelPositions[i], posProbs[i]))
                }
            }
        }
    }

    return result;
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

function toggleMouseZoom() {
    if (zoomMouseData) {
        zoomMouseData = false
        mouseDataZoomButton.innerText = "zoom view angle"
    }
    else {
        zoomMouseData = true
        mouseDataZoomButton.innerText = "unzoom view angle"
    }
    drawTick(null)
}

function toggleShowDistribution() {
    if (showDistribution) {
        showDistribution = false
        mouseDataZoomButton.innerText = "show distribution"
    }
    else {
        showDistribution = true
        showDistributionButton.innerText = "show labels"
    }
    drawTick(null)
}

let playerToEventIndex: Map<number, number> = new Map()
let lastEvent = ""

export function updateEventIdAndSelector(tickData: TickRow) {
    const parser = gameData.parsers.get(curEvent)
    if (curEvent == "none" || !parser.havePlayerLabels) {
        eventIdLabel.style.display = "none"
        eventIdSelector.style.display = "none"
        eventIdSelector.options.length = 0
        eventIdSelector.options.add(new Option("default", "-1"))
        mouseDataDisplayButton.style.display = "none"
        mouseDataZoomButton.style.display = "none"
    }
    else {
        eventIdLabel.style.display = "inline"
        eventIdSelector.style.display = "inline-block"
        const index = getTickToOtherTableIndex(gameData, curEvent)
        const eventArray = gameData.tables.get(curEvent)
        const eventsForTick = index.search([tickData.id, tickData.id])
        let newPlayersTable = gameData.playersTable.length > 0 &&
            !playerToEventIndex.has(gameData.playersTable[0].gameId)
        if (eventIdSelector.options.length == 0 || newPlayersTable || curEvent != lastEvent) {
            lastEvent = curEvent
            eventIdSelector.options.length = 0
            eventIdSelector.options.add(new Option("default", "-1"))
            for (let i = 0 ; i < gameData.playersTable.length; i++) {
                eventIdSelector.options.add(new Option(
                    gameData.playersTable[i].name + "(" + gameData.playersTable[i].id.toString() + ")", "-2"))
                playerToEventIndex.set(gameData.playersTable[i].id, i + 1)
            }
        }
        for (let i = 1; i < eventIdSelector.length; i++) {
            eventIdSelector.options[i].value = "-2"
        }
        for (let i = 0; i < eventsForTick.length; i++) {
            const curEventRow = eventArray[eventsForTick[i]]
            let eventName = curEventRow.otherColumnValues[parser.playersToLabelColumn]
            let sourcePlayerId = parseInt(eventName.split(";")[0])
            eventIdSelector.options[playerToEventIndex.get(sourcePlayerId)].value = curEventRow.id.toString()
        }
        // if old selected event id no longer value, just go with default setting
        selectedEventId = parseInt(eventIdSelector.value)
        if (parser.havePerTickAimTable) {
            mouseDataDisplayButton.style.display = "inline-block"
        }
        if (parser.havePerTickAimTable && parser.havePerTickAimPredictionTable) {
            mouseDataZoomButton.style.display = "inline-block"
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
    mouseDataZoomButton = document.querySelector<HTMLButtonElement>("#event-mouse-zoom")
    mouseDataZoomButton.addEventListener("click", toggleMouseZoom)
    showDistributionButton = document.querySelector<HTMLButtonElement>("#event-show-distribution")
    showDistributionButton.addEventListener("click", toggleShowDistribution)
}