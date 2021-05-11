import {
    GameData,
    getTickToOtherTableIndex,
    Index,
    PlayerAtTickRow,
    playerAtTickTableName,
    Row,
    TickRow,
} from "../data/tables";

let eventSelector: HTMLSelectElement = null
let eventDiv: HTMLDivElement = null
export let curEvent: string = "none"

function basicPlayerText(gameData: GameData, tickData: TickRow,
                         playerIndex: number): string {
    if (gameData.getPlayers(tickData)[playerIndex].isAlive) {
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
    const players: PlayerAtTickRow[] = gameData.getPlayers(tickData)
    // if no event, do nothing special
    if (curEvent == "none" || !index.has(tickData.id)) {
        for (let p = 0; p < players.length; p++) {
            result.push(basicPlayerText(gameData, tickData, p))
        }
        if (result.length < 10) {
            console.log("exit 1")
            console.log(result)
            console.log(gameData)
        }
        return result
    }

    // if event, get min key player number for each player
    const eventArray = gameData.tables.get(curEvent)
    const eventsForTick = index.get(tickData.id)
    for (let p = 0; p < players.length; p++) {
        result.push(basicPlayerText(gameData, tickData, p))
    }

    let numKeyPlayers = eventArray[0].parser.keyPlayerColumns.length;
    let minKeyPlayerNumbers: number[] = [];
    const bogusMinKeyPlayerNumber = 100
    for (let p = 0; p < players.length; p++) {
        minKeyPlayerNumbers[p] = bogusMinKeyPlayerNumber
    }
    for (let eIndex = 0; eIndex < eventsForTick.length; eIndex++)  {
        for (let eIndex = 0; eIndex < eventsForTick.length; eIndex++) {
            let event = eventArray[eventsForTick[eIndex]]
            for (let keyPlayerIndex = 0; keyPlayerIndex < numKeyPlayers;
                 keyPlayerIndex++) {
                const playerIndex =
                    gameData.getPlayerIndex(event.foreignKeyValues[keyPlayerIndex]);
                minKeyPlayerNumbers[playerIndex] =
                    Math.min(minKeyPlayerNumbers[playerIndex], keyPlayerIndex);
            }
        }
    }
    for (let p = 0; p < players.length; p++) {
        if (minKeyPlayerNumbers[p] < bogusMinKeyPlayerNumber) {
            result[p] = minKeyPlayerNumbers[p].toString()
        }
    }
    return result
}

export function setEventText(tickData: TickRow, gameData: GameData) {
    if (curEvent == "none") {
        return
    }
    eventDiv.innerHTML = ""
    const index = getTickToOtherTableIndex(gameData, curEvent)
    if (index.has(tickData.id)) {
        const events = index.get(tickData.id)
        const eventArray = gameData.tables.get(curEvent)
        for (let eIndex = events.length - 1; eIndex >= 0; eIndex--) {
            eventDiv.innerHTML += eventArray[events[eIndex]].getHTML()
        }
    }
}

export function setEventToDraw() {
    curEvent = eventSelector.value
}

export function setupEventDrawing() {
    eventSelector = document.querySelector<HTMLSelectElement>("#event-type")
    curEvent = eventSelector.value;
    eventDiv = document.querySelector<HTMLDivElement>("#events")
}