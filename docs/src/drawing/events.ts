import {
    GameData,
    getTickToOtherTableIndex,
    PlayerAtTickRow,
    playerAtTickTableName, PlayerRow,
    Row,
    TickRow,
} from "../data/tables";
import IntervalTree from "@flatten-js/interval-tree";
import {gameData} from "../data/data";

let eventSelector: HTMLSelectElement = null
let eventDiv: HTMLDivElement = null
let overlaySelector: HTMLSelectElement = null
export let curEvent: string = "none"
export let curOverlay: string = "none"

function basicPlayerText(gameData: GameData, tickData: TickRow,
                         playerIndex: number): string {
    if (gameData.getPlayersAtTick(tickData)[playerIndex].isAlive) {
        return "o"
    }
    else {
        return "x"
    }
}

export function getPlayersText(tickData: TickRow, gameData: GameData): string[] {
    // would be great if could draw more than 1 player
    // but 2 players can shoot each other same tick and not able to visualize that right now
    let result: string[] = []
    const index = getTickToOtherTableIndex(gameData, curEvent)
    const players: PlayerAtTickRow[] = gameData.getPlayersAtTick(tickData)
    // if no event, do nothing special
    if (curEvent == "none" || !index.intersect_any([tickData.demoTickNumber, tickData.demoTickNumber]) ||
        !gameData.parsers.get(curEvent).havePlayerLabels) {
        for (let p = 0; p < players.length; p++) {
            result.push(basicPlayerText(gameData, tickData, p))
        }
        return result
    }

    // if event, get min key player number for each player
    const eventArray = gameData.tables.get(curEvent)
    const eventsForTick = index.search([tickData.demoTickNumber, tickData.demoTickNumber])
    const firstEvent = eventArray[Math.min(...eventsForTick)]
    const parser = gameData.parsers.get(curEvent)
    const playersToLabel = firstEvent.otherColumnValues[parser.playersToLabelColumn].split(";").map(x => parseInt(x))
    const playerLabelIndices = firstEvent.otherColumnValues[parser.playerLabelIndicesColumn].split(";").map(x => parseInt(x))
    for (let p = 0; p < players.length; p++) {
        const indexOfPlayerInLabelsArray = playersToLabel.indexOf(players[p].playerId)
        if (players[p].isAlive && indexOfPlayerInLabelsArray != -1) {
            result.push(parser.playerLabels[playerLabelIndices[indexOfPlayerInLabelsArray]])
        }
        else {
            result.push(basicPlayerText(gameData, tickData, p))
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
        if (index.intersect_any([tickData.demoTickNumber, tickData.demoTickNumber])) {
            const events = index.search([tickData.demoTickNumber, tickData.demoTickNumber])
            const eventArray = gameData.tables.get(curEvent)
            for (let eIndex = events.length - 1; eIndex >= 0; eIndex--) {
                eventDiv.innerHTML += eventArray[events[eIndex]].getHTML()
            }
        }
    }
}

export function setSelectionsToDraw() {
    curEvent = eventSelector.value
    curOverlay = overlaySelector.value;
}

export function setupEventDrawing() {
    eventSelector = document.querySelector<HTMLSelectElement>("#event-type")
    curEvent = eventSelector.value;
    eventDiv = document.querySelector<HTMLDivElement>("#events")
    overlaySelector = document.querySelector<HTMLSelectElement>("#overlay-type")
    curOverlay = overlaySelector.value;
}