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

function basicPlayerText(gameData: GameData, tickId: number,
                         playerIndex: number): string {
    const playerAtTickId = gameData.ticksToOtherTablesIndices
        .get(playerAtTickTableName).get(tickId)[playerIndex]
    if (gameData.playerAtTicksTable[playerAtTickId].isAlive) {
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
            result.push(basicPlayerText(gameData, tickData.id, p))
        }
        if (result.length < 10) {
            console.log("exit 1")
            console.log(result)
            console.log(gameData)
        }
        return result
    }

    // if event, print source and target if present
    const eventArray = gameData.tables.get(curEvent)
    const eventsForTick = index.get(tickData.id)
    for (let p = 0; p < players.length; p++) {
        // print every player that is a target or source
        if (eventArray[0].getSource !== undefined ||
            eventArray[0].getTargets !== undefined) {
            let isTargetOrSource = false
            for (let eIndex = 0; eIndex < eventsForTick.length; eIndex++)  {
                let event = eventArray[eventsForTick[eIndex]]
                // targets can be undefined even if source is defined
                if (eventArray[0].getTargets !== undefined &&
                    event.getTargets().includes(tickdata.players[p].name)) {
                    result.push("t")
                    isTargetOrSource = true
                    break;
                }
                else if (eventArray[0].getSource !== undefined &&
                    event.getSource() === tickdata.players[p].name) {
                    result.push("s")
                    isTargetOrSource = true
                    break;
                }
            }
            if (!isTargetOrSource) {
                result.push(basicPlayerText(tickdata, p))
            }
        }
        else {
            result.push(basicPlayerText(tickdata, p))
        }
    }
    if (result.length < 10) {
        console.log("exit 2")
        console.log(result)
        console.log(gameData)
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