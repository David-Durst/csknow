import {gameData, initialized} from "../data/data"
import {
    filteredData,
    clearFilterData,
    filterRegion, stopFilteringEvents
} from "../controller/filter";
import {PositionRow} from "../data/tables";
import {getPackedSettings} from "http2";
import {
    getPlayersText,
    setEventText, setEventToDraw,
    setupEventDrawing,
} from "./events";
import {clearCustomFilter} from "../controller/ide_filters";
import {getCurTickIndex, setTickLabel} from "../controller/tickSelector";
import {start} from "repl";
import {match} from "assert";

export const d2_top_left_x = -2476
export const d2_top_left_y = 3239
export const canvasWidth = 700
export const canvasHeight = 700
export const minimapWidth = 1024
export const minimapHeight = 1024
export const minimapScale = 4.4
export let canvas: HTMLCanvasElement = null;
export let ctx: CanvasRenderingContext2D = null;
export const minimap = new Image();
minimap.src = "de_dust2_radar_spectate.png";
let xMapLabel: HTMLLabelElement = null;
let yMapLabel: HTMLLabelElement = null;
let xCanvasLabel: HTMLLabelElement = null;
let yCanvasLabel: HTMLLabelElement = null;
let tScoreLabel: HTMLLabelElement = null;
let ctScoreLabel: HTMLLabelElement = null;
let playerNameLabel: HTMLLabelElement = null;
let playerCopyButton: HTMLButtonElement = null;
let playerCopyText: HTMLInputElement = null;
let configClientButton: HTMLAnchorElement = null;
let minZ = 0;
let maxZ = 0;
const background = new Image();
background.src = "de_dust2_radar_spectate.png";
const black = "rgba(0,0,0,1.0)";
const gray = "rgba(159,159,159,1.0)";
const dark_blue = "rgba(4,190,196,1.0)";
const light_blue = "rgba(194,255,243,1.0)";
const purple = "rgb(160,124,205)";
const dark_red = "rgba(209,0,0,1.0)";
const light_red = "rgba(255,143,143,1.0)";
const yellow = "rgb(252,198,102)";
const green = "rgba(0,150,0,1.0)";

let demoURL: string = ""
export function setDemoURL(newUrl: string) {
    demoURL = newUrl
}

let demoName: string = ""
export function setDemoName(newName: string) {
    demoName = newName
}

// see last post by randunel and csgo/resources/overview/de_dust2.txt
// https://forums.alliedmods.net/showthread.php?p=2690857#post2690857
class MapCoordinate {
    x: number
    y: number

    constructor(x: number, y: number, fromCanvasPixels: boolean) {
        if (fromCanvasPixels) {
            const pctX = x / canvasWidth;
            this.x = d2_top_left_x + minimapScale * minimapWidth * pctX
            const pctY = y / canvasHeight;
            this.y = d2_top_left_y - minimapScale * minimapHeight * pctY
        } else {
            this.x = x
            this.y = y
        }
    }

    getCanvasX() {
        return (this.x - d2_top_left_x) /
            (minimapScale * minimapWidth) * canvasWidth
    }

    getCanvasY() {
        return (d2_top_left_y - this.y) /
            (minimapScale * minimapHeight) * canvasHeight
    }
}

function getMouseCoordinate(e: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    const xCanvas = (e.clientX - rect.left)
    const yCanvas = (e.clientY - rect.top)
    return new MapCoordinate(xCanvas, yCanvas, true)
}

let drawingRegionFilter: boolean = false
let definedRegionFilter: boolean = false
let emptyFilter: boolean = false
let topLeftCoordinate: MapCoordinate = null
let bottomRightCoordinate: MapCoordinate = null

function startingRegionFilter(e: MouseEvent) {
    if (!initialized) {
        return
    }
    clearFilterData()
    drawingRegionFilter = true
    definedRegionFilter = false
    emptyFilter = false
    topLeftCoordinate = getMouseCoordinate(e)
    bottomRightCoordinate = topLeftCoordinate
    drawTick(null)
}

function finishedRegionFilter(e: MouseEvent) {
    if (!initialized) {
        return
    }
    drawingRegionFilter = false
    definedRegionFilter = true
    bottomRightCoordinate = getMouseCoordinate(e)
    const filterTopLeftX = topLeftCoordinate.x
    const filterTopLeftY = topLeftCoordinate.y
    const filterBottomRightX = bottomRightCoordinate.x
    const filterBottomRightY = bottomRightCoordinate.y
    const minX = Math.min(filterTopLeftX, filterBottomRightX)
    const minY = Math.min(filterTopLeftY, filterBottomRightY)
    const maxX = Math.max(filterTopLeftX, filterBottomRightX)
    const maxY = Math.max(filterTopLeftY, filterBottomRightY)
    emptyFilter = !filterRegion(minX, minY, maxX, maxY)
    drawTick(null)
}

function clearFilterButton() {
    if (!initialized) {
        return
    }
    stopFilteringEvents()
    clearCustomFilter()
    clearFilterData()
    drawingRegionFilter = false
    definedRegionFilter = false
    drawTick(null)
}

function setPosition(playerX: number, playerY: number, playerZ: number) {
    const startX = playerX - 16;
    const endX = playerX + 16;
    const startY = playerY - 16;
    const endY = playerY + 16;
    const startZ = playerZ;
    const endZ = playerZ + 72;
    playerCopyText.value = "box " + startX.toString() + " " +
        startY.toString() + " " + startZ.toString() + " " +
        endX.toString() + " " + endY.toString() + " " + endZ.toString()
}

function copyPosition() {
    playerCopyText.select();
    document.execCommand("copy");
}

let selectedPlayer = -1
function trackMouse(e: MouseEvent) {
    if (!initialized) {
        return
    }
    const minimapCoordinate = getMouseCoordinate(e)
    if (drawingRegionFilter) {
        bottomRightCoordinate = minimapCoordinate
    }
    xMapLabel.innerHTML = minimapCoordinate.x.toPrecision(6)
    yMapLabel.innerHTML = minimapCoordinate.y.toPrecision(6)
    xCanvasLabel.innerHTML = minimapCoordinate.getCanvasX().toPrecision(6)
    yCanvasLabel.innerHTML = minimapCoordinate.getCanvasY().toPrecision(6)

    const curTickIndex = getCurTickIndex()
    const tickData: PositionRow = filteredData.position[curTickIndex]
    for (let p = 0; p < tickData.players.length; p++) {
        const playerCoordinate = new MapCoordinate(
            tickData.players[p].xPosition, tickData.players[p].yPosition, false)
        if (playerCoordinate.getCanvasX() - 10 <= minimapCoordinate.getCanvasX() &&
            playerCoordinate.getCanvasX() + 10 >= minimapCoordinate.getCanvasX() &&
            playerCoordinate.getCanvasY() - 10 <= minimapCoordinate.getCanvasY() &&
            playerCoordinate.getCanvasY() + 10 >= minimapCoordinate.getCanvasY()) {
            playerNameLabel.innerHTML = tickData.players[p].name
            setPosition(tickData.players[p].xPosition,
                tickData.players[p].yPosition,
                tickData.players[p].zPosition)
            selectedPlayer = p;
            drawTick(null)
            return
        }
    }
    drawTick(null)
}

export function drawTick(e: InputEvent) {
    ctx.drawImage(minimap,0,0,minimapWidth,minimapHeight,0,0,
        canvasWidth,canvasHeight);
    ctx.textBaseline = "middle"
    ctx.textAlign = "center"
    const curTickIndex = getCurTickIndex()
    setTickLabel(filteredData.position[curTickIndex].demoTickNumber,
        filteredData.position[curTickIndex].gameTickNumber)
    const tickData: PositionRow = filteredData.position[curTickIndex]
    tScoreLabel.innerHTML = tickData.tScore.toString()
    ctScoreLabel.innerHTML = tickData.ctScore.toString()
    let playersText = getPlayersText(tickData, filteredData)
    for (let p = 0; p < tickData.players.length; p++) {
        let playerText = playersText[p]
        ctx.fillStyle = dark_blue
        if (tickData.players[p].team == 3) {
            if (p == selectedPlayer) {
                ctx.fillStyle = purple
            }
            else if (playerText == "t" || playerText == "s") {
                ctx.fillStyle = light_blue
            }
        }
        else {
            ctx.fillStyle = dark_red
            if (p == selectedPlayer) {
                ctx.fillStyle = yellow
            }
            else if (playerText == "t" || playerText == "s") {
                ctx.fillStyle = light_red
            }
        }
        const location = new MapCoordinate(
            tickData.players[p].xPosition,
            tickData.players[p].yPosition,
            false);
        const zScaling = (tickData.players[p].zPosition - minZ) / (maxZ - minZ)
        ctx.font = (zScaling * 20 + 30).toString() + "px Arial"
        ctx.fillText(playerText, location.getCanvasX(), location.getCanvasY())
        ctx.save()
        ctx.translate(location.getCanvasX(), location.getCanvasY())
        ctx.rotate((90-tickData.players[p].xViewDirection)/180*Math.PI)
        // divide by -90 as brighter means up and < 0 is looking up
        const yNeg1To1 = tickData.players[p].yViewDirection / -90 
        const yLogistic = 2 / (1 + Math.pow(Math.E, -8 * yNeg1To1))
        ctx.filter = "brightness(" + yLogistic + ")"
        if (tickData.players[p].isAlive) {
            //ctx.fillText("^", 0, 0)
            ctx.fillRect(-2, -13 + -7 * zScaling, 4, 10)
        }
        ctx.restore()
        //ctx.fillRect(location.getCanvasX(), location.getCanvasY(), 1, 1)
    }
    if (drawingRegionFilter || definedRegionFilter) {
        if (emptyFilter) {
            ctx.strokeStyle = dark_red
        }
        else {
            ctx.strokeStyle = green
        }
        ctx.lineWidth = 3.0
        ctx.strokeRect(topLeftCoordinate.getCanvasX(), topLeftCoordinate.getCanvasY(),
            bottomRightCoordinate.getCanvasX() - topLeftCoordinate.getCanvasX(),
            bottomRightCoordinate.getCanvasY() - topLeftCoordinate.getCanvasY())
    }
    setEventText(tickData, filteredData)
    // setup client config for this tick
    let startDemoTick = filteredData.position[curTickIndex].gameTickNumber - 100
    startDemoTick = Math.max(startDemoTick, 10)
    configClientButton.href = URL.createObjectURL(new Blob(
        // https://csgo-ranks.com/demo-commands/
        ["//" + demoURL + "\n//" + demoName + "\nplaydemo " + demoName +
        "\ndemo_gototick " + startDemoTick.toString() + "\ndemo_pause" +
        "\ndemo_timescale 0.125\nspec_player_by_name " + playerNameLabel.innerHTML],
        {type: 'text/plain'}
    ))
}
export let maxViewY = 0
export let maxViewYT = 0
export let maxViewYP = 0
export function setupMatchDrawing() {
    const numTicks = gameData.position.length
    for (let t = 0; t < numTicks; t++) {
        for (let p = 0; p < gameData.position[t].players.length; p++) {
            minZ = Math.min(minZ, gameData.position[t].players[p].zPosition)
            maxZ = Math.max(maxZ, gameData.position[t].players[p].zPosition)
            if (gameData.position[t].gamePhase !== 5 &&
                gameData.position[t].gamePhase !== 1 &&
                gameData.position[t].gamePhase !== 4 &&
                !gameData.position[t].isWarmup) {
                maxViewY = Math.max(maxViewY, gameData.position[t].players[p].yViewDirection)
                maxViewYT = t
                maxViewYP = p
            }
        }
    }
    drawTick(null)
}

export function setupCanvas() {
    canvas = <HTMLCanvasElement> document.querySelector("#myCanvas")
    ctx = canvas.getContext('2d')
    xMapLabel = document.querySelector<HTMLLabelElement>("#xposMap")
    yMapLabel = document.querySelector<HTMLLabelElement>("#yposMap")
    xCanvasLabel = document.querySelector<HTMLLabelElement>("#xposCanvas")
    yCanvasLabel = document.querySelector<HTMLLabelElement>("#yposCanvas")
    tScoreLabel = document.querySelector<HTMLLabelElement>("#t_score")
    ctScoreLabel = document.querySelector<HTMLLabelElement>("#ct_score")
    playerNameLabel = document.querySelector<HTMLLabelElement>("#playerName")
    playerCopyText = document.querySelector<HTMLInputElement>("#copy_text")
    playerCopyButton = document.querySelector<HTMLButtonElement>("#copy_button")
    playerCopyButton.addEventListener("click", copyPosition)
    configClientButton = document.querySelector<HTMLAnchorElement>("#download_client")
    canvas.addEventListener("mousemove", trackMouse)
    canvas.addEventListener("mousedown", startingRegionFilter)
    canvas.addEventListener("mouseup", finishedRegionFilter)
    setupEventDrawing()
}

function setEventsAndRedraw() {
    setEventToDraw()
    drawTick(null)
}

export function setupCanvasHandlers() {
    document.querySelector<HTMLInputElement>("#tick-selector").addEventListener("input", drawTick)
    document.querySelector<HTMLButtonElement>("#clear_filter").addEventListener("click", clearFilterButton)
    document.querySelector<HTMLSelectElement>("#event-type").addEventListener("change", setEventsAndRedraw)
}