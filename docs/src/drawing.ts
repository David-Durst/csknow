import {gameData, initialized, PositionRow} from "./data"
import {filteredData, clearRegionFilterData} from "./filter";

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
export let tickSelector: HTMLInputElement = null;
export let tickLabel: HTMLLabelElement = null;
let tScoreLabel: HTMLLabelElement = null;
let ctScoreLabel: HTMLLabelElement = null;
let playerNameLabel: HTMLLabelElement = null;
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
let topLeftCoordinate: MapCoordinate = null
let bottomRightCoordinate: MapCoordinate = null

function startingRegionFilter(e: MouseEvent) {
    if (!initialized) {
        return
    }
    drawingRegionFilter = true
    definedRegionFilter = false
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
    //filterRegion(minX, minY, maxX, maxY)
    drawTick(null)
}

function clearRegionFilterDrawing() {
    if (!initialized) {
        return
    }
    clearRegionFilterData()
    drawingRegionFilter = false
    definedRegionFilter = false
    drawTick(null)
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

    const curTick = parseInt(tickSelector.value)
    const tickData: PositionRow = filteredData.position[curTick]
    for (let p = 0; p < tickData.players.length; p++) {
        const playerCoordinate = new MapCoordinate(
            tickData.players[p].xPosition, tickData.players[p].yPosition, false)
        if (playerCoordinate.getCanvasX() <= minimapCoordinate.getCanvasX() &&
            playerCoordinate.getCanvasX() + 20 >= minimapCoordinate.getCanvasX() &&
            playerCoordinate.getCanvasY() - 20 <= minimapCoordinate.getCanvasY() &&
            playerCoordinate.getCanvasY() >= minimapCoordinate.getCanvasY()) {
            playerNameLabel.innerHTML = tickData.players[p].name
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
    const curTick = parseInt(tickSelector.value)
    tickLabel.innerHTML = filteredData.position[curTick].tickNumber.toString()
    const tickData: PositionRow = filteredData.position[curTick]
    tScoreLabel.innerHTML = tickData.tScore.toString()
    ctScoreLabel.innerHTML = tickData.ctScore.toString()
    for (let p = 0; p < tickData.players.length; p++) {
        let playerText = "o"
        if (!tickData.players[p].isAlive) {
            playerText = "x"
        }
        ctx.fillStyle = dark_blue
        if (p == selectedPlayer) {
            ctx.fillStyle = purple
        }
        if (tickData.players[p].team == 2) {
            ctx.fillStyle = dark_red
            if (p == selectedPlayer) {
                ctx.fillStyle = yellow
            }
        }
        const location = new MapCoordinate(
            tickData.players[p].xPosition,
            tickData.players[p].yPosition,
            false);
        const zScaling = (tickData.players[p].zPosition - minZ) / (maxZ - minZ)
        ctx.font = (zScaling * 20 + 30).toString() + "px Arial"
        ctx.fillText(playerText, location.getCanvasX(), location.getCanvasY())
        //ctx.fillRect(location.getCanvasX(), location.getCanvasY(), 1, 1)
    }
    if (drawingRegionFilter || definedRegionFilter) {
        ctx.strokeStyle = green
        ctx.lineWidth = 3.0
        ctx.strokeRect(topLeftCoordinate.getCanvasX(), topLeftCoordinate.getCanvasY(),
            bottomRightCoordinate.getCanvasX() - topLeftCoordinate.getCanvasX(),
            bottomRightCoordinate.getCanvasY() - topLeftCoordinate.getCanvasY())
    }
}

export function setupMatch() {
    const numTicks = gameData.position.length
    for (let t = 0; t < numTicks; t++) {
        for (let p = 0; p < gameData.position[t].players.length; p++) {
            minZ = Math.min(minZ, gameData.position[t].players[p].zPosition)
            maxZ = Math.max(maxZ, gameData.position[t].players[p].zPosition)
        }
    }
    tickSelector.max = (gameData.position.length - 1).toString()
    drawTick(null)
}

export function setupCanvas() {
    canvas = <HTMLCanvasElement> document.querySelector("#myCanvas")
    ctx = canvas.getContext('2d')
    tickSelector = document.querySelector<HTMLInputElement>("#tick-selector")
    tickLabel = document.querySelector<HTMLLabelElement>("#cur-tick")
    xMapLabel = document.querySelector<HTMLLabelElement>("#xposMap")
    yMapLabel = document.querySelector<HTMLLabelElement>("#yposMap")
    xCanvasLabel = document.querySelector<HTMLLabelElement>("#xposCanvas")
    yCanvasLabel = document.querySelector<HTMLLabelElement>("#yposCanvas")
    tScoreLabel = document.querySelector<HTMLLabelElement>("#t_score")
    ctScoreLabel = document.querySelector<HTMLLabelElement>("#ct_score")
    playerNameLabel = document.querySelector<HTMLLabelElement>("#playerName")
    canvas.addEventListener("mousemove", trackMouse)
    canvas.addEventListener("mousedown", startingRegionFilter)
    canvas.addEventListener("mouseup", finishedRegionFilter)
    document.querySelector<HTMLInputElement>("#tick-selector").addEventListener("input", drawTick)
    document.querySelector<HTMLButtonElement>("#clear_filter").addEventListener("click", clearRegionFilterDrawing)
    tickLabel.innerHTML = "0"
}