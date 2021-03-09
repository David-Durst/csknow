import {
    DemoData,
    GameData,
    getEventArray,
    getEventIndex,
    PositionRow, Printable
} from "../data/tables";

let eventSelector: HTMLSelectElement = null
let curEvent: string = "none"

class Event_Status {
    event: DemoData
    source: string
    targets: string[]
}

function basicPlayerText(tickdata: PositionRow, playerIndex: number): string {
    if (tickdata.players[playerIndex].isAlive) {
        return "o"
    }
    else {
        return "x"
    }
}

export function getPlayersText(tickdata: PositionRow, gameData: GameData,
                               selectedPlayer: string): string[] {
    // would be great if could draw more than 1 player
    // but 2 players can shoot each other same tick and not able to visualize that right now
    let result: string[] = []
    // if no event, do nothing special
    if (curEvent == "none") {
        for (let p = 0; p < tickdata.players.length; p++) {
            result.push(basicPlayerText(tickdata, p))
        }
        return result
    }

    // if event, print source and target if present
    const index = getEventIndex(gameData, curEvent)
    if (index.has(tickdata.demoTickNumber)) {
        const eventArray = getEventArray(gameData, curEvent)
        const eventsForTick = index.get(tickdata.demoTickNumber)
        for (let p = 0; p < tickdata.players.length; p++) {
            // print every player that is a target of the source that is selectedPlayer
            if (tickdata.players[p].name != selectedPlayer) {
                if (eventArray[0].getTargets !== undefined &&
                    eventArray[0].getSource !== undefined) {
                    let isTarget = false
                    for (let eIndex = 0; eIndex < eventsForTick.length; eIndex++)  {
                        let event = eventArray[eventsForTick[eIndex]]
                        if (event.getSource() === selectedPlayer &&
                            event.getTargets().includes(tickdata.players[p].name)) {
                            result.push("t")
                            isTarget = true
                            break;
                        }
                    }
                    if (!isTarget) {
                        result.push(basicPlayerText(tickdata, p))
                    }
                }
                else {
                    result.push(basicPlayerText(tickdata, p))
                }
            }
            // print the source
            else {
                if (eventArray[0].getSource !== undefined) {
                    let isSource = false
                    for (let eIndex = 0; eIndex < eventsForTick.length; eIndex++) {
                        let event = eventArray[eventsForTick[eIndex]]
                        if (event.getSource() === tickdata.players[p].name) {
                            result.push("s")
                            isSource = true
                            break;
                        }
                    }
                    if (!isSource) {
                        result.push(basicPlayerText(tickdata, p))
                    }
                }
            }
        }
        return result
    }
}

export function setEventToDraw() {
    curEvent = eventSelector.value
}

export function setupEventDrawing() {
    eventSelector = document.querySelector<HTMLSelectElement>("#event-type")
    document.querySelector<HTMLSelectElement>("#download-type").addEventListener("change", setEventToDraw)
    curEvent = eventSelector.value;
}