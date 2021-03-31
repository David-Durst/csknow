import {
    DemoData,
    GameData,
    getEventArray,
    getEventIndex,
    PositionRow, Printable
} from "../data/tables";

let eventSelector: HTMLSelectElement = null
let eventDiv: HTMLDivElement = null
export let curEvent: string = "none"

function basicPlayerText(tickdata: PositionRow, playerIndex: number): string {
    if (tickdata.players[playerIndex].isAlive) {
        return "o"
    }
    else {
        return "x"
    }
}

export function getPlayersText(tickdata: PositionRow, gameData: GameData): string[] {
    // would be great if could draw more than 1 player
    // but 2 players can shoot each other same tick and not able to visualize that right now
    let result: string[] = []
    const index = getEventIndex(gameData, curEvent)
    // if no event, do nothing special
    if (curEvent == "none" || !index.has(tickdata.demoTickNumber)) {
        for (let p = 0; p < tickdata.players.length; p++) {
            result.push(basicPlayerText(tickdata, p))
        }
        if (result.length < 10) {
            console.log("exit 1")
            console.log(result)
            console.log(gameData)
        }
        return result
    }

    // if event, print source and target if present
    const eventArray = getEventArray(gameData, curEvent)
    const eventsForTick = index.get(tickdata.demoTickNumber)
    for (let p = 0; p < tickdata.players.length; p++) {
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

export function setEventText(tickdata: PositionRow, gameData: GameData) {
    if (curEvent == "none") {
        return
    }
    eventDiv.innerHTML = ""
    const index = getEventIndex(gameData, curEvent)
    if (index.has(tickdata.demoTickNumber)) {
        const events = index.get(tickdata.demoTickNumber)
        const eventArray = getEventArray(gameData, curEvent)
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