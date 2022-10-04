import {gameData, initialized} from "../data/data"
import {
    filteredData,
    applyJustEventFilter,
    filterRegion, stopFilteringEvents
} from "../controller/filter";
import {parseBool, PlayerAtTickRow, PlayerRow, TickRow} from "../data/tables";
import {getPackedSettings} from "http2";
import {
    activeEvent,
    curOverlay, DEFAULT_ALIVE_STRING,
    getPlayersText, setEventsOverlaysToDraw,
    setEventText, setupEventDrawing, updateEventIdAndSelector,
} from "./events";
import {clearCustomFilter} from "../controller/ide_filters";
import {getCurTickIndex, setTickLabel} from "../controller/selectors";
import {start} from "repl";
import {match} from "assert";
import {createCharts, drawMouseData} from "./mouseDrawing";

export const d2_top_left_x = -2476
export const d2_top_left_y = 3239
// size of image when drawing on website
export const defaultCanvasSize = 700
export const bigCanvasSize = 2048
export let canvasWidth = defaultCanvasSize
export let canvasHeight = defaultCanvasSize
// size of image from file
export const minimapWidth = 2048
export const minimapHeight = 2048
// conversion from minimap coordinates to csgo coordinates I figured out to be 4.4
// when the input file was 1024x1024
export const minimapScale = 4.4 * 1024 / minimapHeight
let fontScale = 1.0
export let mainCanvas: HTMLCanvasElement = null;
export let mainCtx: CanvasRenderingContext2D = null;
let cacheGridCanvas: HTMLCanvasElement = null;
let cacheGridCtx: CanvasRenderingContext2D = null;
let cacheTargetCanvas: HTMLCanvasElement = null;
let cacheTargetCtx: CanvasRenderingContext2D = null;
let lastCacheOverlay: string = null;
export let kymographCanvas: HTMLCanvasElement = null;
export let kymographCtx: CanvasRenderingContext2D = null;
export let scatterCanvas: HTMLCanvasElement = null;
export let scatterCtx: CanvasRenderingContext2D = null;
export let inferenceCanvas: HTMLCanvasElement = null;
export let inferenceCtx: CanvasRenderingContext2D = null;
export const minimap = new Image();
minimap.src = "vis_images/de_dust2_radar_upsampled_all_labels.png";
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
let toggleCanvasSizeButton: HTMLButtonElement = null;
let minZ = 0;
let maxZ = 0;
const black = "rgba(0,0,0,1.0)";
const gray = "rgba(159,159,159,1.0)";
const dark_blue = "rgba(4,190,196,1.0)";
const light_blue = "rgba(194,255,243,1.0)";
const purple = "rgb(160,124,205)";
const dark_red = "rgba(209,0,0,1.0)";
const light_red = "rgba(255,143,143,1.0)";
const yellow = "rgb(252,198,102)";
const green = "rgba(0,150,0,1.0)";
export let smallMode: boolean = false

let demoURL: string = ""
export function setDemoURL(newUrl: string) {
    demoURL = newUrl
}

let demoName: string = ""
export function setDemoName(newName: string) {
    demoName = newName
}

export function toggleCanvasSize() {
    if (canvasWidth == defaultCanvasSize) {
        resizeCanvas(bigCanvasSize)
    }
    else {
        resizeCanvas(defaultCanvasSize)
    }
    drawTick(null)
}

function resizeCanvas(newSize: number) {
    canvasWidth = newSize
    canvasHeight = newSize
    mainCanvas.width = newSize
    mainCanvas.height = newSize
    cacheGridCanvas.width = newSize
    cacheGridCanvas.height = newSize
    cacheTargetCanvas.width = newSize
    cacheTargetCanvas.height = newSize
    fontScale = newSize / defaultCanvasSize
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
    const rect = mainCanvas.getBoundingClientRect();
    const xCanvas = (e.clientX - rect.left)
    const yCanvas = (e.clientY - rect.top)
    return new MapCoordinate(xCanvas, yCanvas, true)
}

let drawingRegionFilter: boolean = false
let definedRegionFilter: boolean = false
let emptyFilter: boolean = false
let topLeftCoordinate: MapCoordinate = null
let bottomRightCoordinate: MapCoordinate = null

export function initFilterVars() {
    drawingRegionFilter = false
    definedRegionFilter = false
    emptyFilter = false
    topLeftCoordinate = null
    bottomRightCoordinate = null
}

function startingRegionFilter(e: MouseEvent) {
    if (!initialized || smallMode) {
        return
    }
    applyJustEventFilter()
    drawingRegionFilter = true
    definedRegionFilter = false
    emptyFilter = false
    topLeftCoordinate = getMouseCoordinate(e)
    bottomRightCoordinate = topLeftCoordinate
    drawTick(null)
}

function finishedRegionFilter(e: MouseEvent) {
    if (!initialized || smallMode) {
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
    applyJustEventFilter()
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
let secondToLastMousePosition = new MapCoordinate(0, 0, true);
let lastMousePosition = new MapCoordinate(0, 0, true);
function trackMouse(e: MouseEvent) {
    if (!initialized) {
        return
    }
    const minimapCoordinate = getMouseCoordinate(e)
    secondToLastMousePosition = lastMousePosition
    lastMousePosition = minimapCoordinate;
    if (drawingRegionFilter) {
        bottomRightCoordinate = minimapCoordinate
    }
    xMapLabel.innerHTML = minimapCoordinate.x.toPrecision(6)
    yMapLabel.innerHTML = minimapCoordinate.y.toPrecision(6)
    xCanvasLabel.innerHTML = minimapCoordinate.getCanvasX().toPrecision(6)
    yCanvasLabel.innerHTML = minimapCoordinate.getCanvasY().toPrecision(6)

    const curTickIndex = getCurTickIndex()
    const tickData: TickRow = filteredData.ticksTable[curTickIndex]
    const players: PlayerAtTickRow[] = gameData.getPlayersAtTick(tickData)
    for (let p = 0; p < players.length; p++) {
        const playerCoordinate = new MapCoordinate(
            players[p].posX, players[p].posY, false)
        if (playerCoordinate.getCanvasX() - 10 <= minimapCoordinate.getCanvasX() &&
            playerCoordinate.getCanvasX() + 10 >= minimapCoordinate.getCanvasX() &&
            playerCoordinate.getCanvasY() - 10 <= minimapCoordinate.getCanvasY() &&
            playerCoordinate.getCanvasY() + 10 >= minimapCoordinate.getCanvasY()) {
            playerNameLabel.innerHTML = gameData.getPlayerName(players[p].playerId)
            setPosition(players[p].posX, players[p].posY, players[p].posZ)
            selectedPlayer = players[p].playerId;
            drawTick(null)
            return
        }
    }
    drawTick(null)
}

let altPressed: boolean = false
let controlPressed: boolean = false
let shiftPressed: boolean = false
let activeKeys: Set<string> = new Set<string>()
function trackKeyDown(e: KeyboardEvent) {
    activeKeys.add(e.key)
    altPressed = e.altKey
    controlPressed = e.ctrlKey
    shiftPressed = e.shiftKey
    if (controlPressed) {
        drawTick(null)
    }
}

function trackKeyUp(e: KeyboardEvent) {
    activeKeys.delete(e.key)
    altPressed = e.altKey
    controlPressed = e.ctrlKey
    shiftPressed = e.shiftKey
}

// three levels of priority
// level 2: special state for event
// level 1: alive, no special state
// level 0: dead
function assignPriority(pat: PlayerAtTickRow, renderString: string): number {
    if (!pat.isAlive) {
        return 0
    }
    else if (renderString == DEFAULT_ALIVE_STRING) {
        return 1
    }
    else {
        return 2
    }
}

export function drawTick(e: InputEvent) {
    const ftmp = filteredData
    mainCtx.drawImage(minimap,0,0,minimapWidth,minimapHeight,0,0,
        canvasWidth,canvasHeight);
    mainCtx.textBaseline = "middle"
    mainCtx.textAlign = "center"
    cacheTargetCtx.textAlign = "center"
    const curTickIndex = getCurTickIndex()
    setTickLabel(filteredData.ticksTable[curTickIndex].demoTickNumber,
        filteredData.ticksTable[curTickIndex].gameTickNumber)
    const tickData: TickRow = filteredData.ticksTable[curTickIndex]
    tScoreLabel.innerHTML = gameData.getRound(tickData).tWins.toString()
    ctScoreLabel.innerHTML = gameData.getRound(tickData).ctWins.toString()
    // need to update event id before getting players text as need to know current event to set player text
    updateEventIdAndSelector(tickData)
    let playersText = getPlayersText(tickData, filteredData)
    const players = gameData.getPlayersAtTick(tickData)
    drawMouseData(kymographCanvas, scatterCanvas, inferenceCanvas, gameData, tickData, activeEvent)
    // sorting prevents flickering as players drawn in same order always
    // sort by priority, then by id if same priority
    players.sort((a: PlayerAtTickRow, b:PlayerAtTickRow) => {
        let aPriority = assignPriority(a, playersText.get(a.playerId))
        let bPriority = assignPriority(b, playersText.get(b.playerId))
        if (aPriority != bPriority) {
            return aPriority - bPriority
        }
        else {
            return a.playerId - b.playerId
        }
    })
    for (let p = 0; p < players.length; p++) {
        let playerText = playersText.get(players[p].playerId)
        mainCtx.fillStyle = dark_blue
        if (players[p].team == 3) {
            if (players[p].playerId == selectedPlayer) {
                mainCtx.fillStyle = purple
            }
            else if (playerText == "t" || playerText == "s") {
                mainCtx.fillStyle = light_blue
            }
        }
        else {
            mainCtx.fillStyle = dark_red
            if (players[p].playerId == selectedPlayer) {
                mainCtx.fillStyle = yellow
            }
            else if (playerText == "t" || playerText == "s") {
                mainCtx.fillStyle = light_red
            }
        }
        const location = new MapCoordinate(
            players[p].posX,
            players[p].posY,
            false);
        const zScaling = (players[p].posZ - minZ) / (maxZ - minZ)
        mainCtx.font = ((zScaling * 20 + 30) * fontScale).toString() + "px Arial"
        mainCtx.fillText(playerText, location.getCanvasX(), location.getCanvasY())
        mainCtx.save()
        mainCtx.translate(location.getCanvasX(), location.getCanvasY())
        mainCtx.rotate((90-players[p].viewX)/180*Math.PI)
        // divide by -90 as brighter means up and < 0 is looking up
        const yNeg1To1 = players[p].viewY / -90
        const yLogistic = 2 / (1 + Math.pow(Math.E, -8 * yNeg1To1))
        mainCtx.filter = "brightness(" + yLogistic + ")"
        if (players[p].isAlive) {
            //ctx.fillText("^", 0, 0)
            mainCtx.fillRect(-2 * fontScale, (-13 + -7 * zScaling) * fontScale, 4 * fontScale, 10 * fontScale)
        }
        mainCtx.restore()
        //ctx.fillRect(location.getCanvasX(), location.getCanvasY(), 1, 1)
    }
    if (drawingRegionFilter || definedRegionFilter) {
        if (emptyFilter) {
            mainCtx.strokeStyle = dark_red
        }
        else {
            mainCtx.strokeStyle = green
        }
        mainCtx.lineWidth = 3.0
        mainCtx.strokeRect(topLeftCoordinate.getCanvasX(), topLeftCoordinate.getCanvasY(),
            bottomRightCoordinate.getCanvasX() - topLeftCoordinate.getCanvasX(),
            bottomRightCoordinate.getCanvasY() - topLeftCoordinate.getCanvasY())
    }
    if (curOverlay.includes("mesh") || curOverlay.includes("cells")) {
        mainCtx.fillStyle = green
        const overlayRows = filteredData.overlays.get(curOverlay)
        const overlayLabelsRows = filteredData.overlays.get(filteredData.parsers.get(curOverlay).overlayLabelsQuery)
        let connectionAreaIds: number[] = [];
        let targetAreaId = -1
        let targetPlaceName = ""
        let targetX = -1
        let targetY = -1
        let targetFontSize = -1
        // if not already cached
        // draw all area outlines and compute target area
        let drawOutlines = false
        if (lastCacheOverlay == null || lastCacheOverlay != curOverlay) {
            lastCacheOverlay = curOverlay
            drawOutlines = true
            cacheGridCtx.clearRect(0, 0, cacheGridCanvas.width, cacheGridCanvas.height)
        }
        // draw target only after mouse stops moving and in new box, no need to dynamically update it every time
        // remove all movements outside canvas
        let drawTarget = lastMousePosition != secondToLastMousePosition &&
            lastMousePosition.getCanvasX() > 0. &&
            lastMousePosition.getCanvasX() < mainCanvas.width &&
            lastMousePosition.getCanvasY() > 0. &&
            lastMousePosition.getCanvasY() < mainCanvas.height &&
            controlPressed;
        if (drawOutlines || drawTarget) {
            cacheTargetCtx.clearRect(0, 0, cacheTargetCanvas.width, cacheTargetCanvas.height)
        }
        for (let o = 0; (drawOutlines || drawTarget) && o < overlayRows.length; o++) {
            const overlayRow = overlayRows[o]
            const minCoordinate = new MapCoordinate(
                parseFloat(overlayRow.otherColumnValues[2]),
                parseFloat(overlayRow.otherColumnValues[3]),
                false);
            const maxCoordinate = new MapCoordinate(
                parseFloat(overlayRow.otherColumnValues[5]),
                parseFloat(overlayRow.otherColumnValues[6]),
                false);
            const avgX = (minCoordinate.getCanvasX() + maxCoordinate.getCanvasX()) / 2
            const avgY = (minCoordinate.getCanvasY() + maxCoordinate.getCanvasY()) / 2
            const avgZ = (parseFloat(overlayRow.otherColumnValues[4]) + parseFloat(overlayRow.otherColumnValues[7])) / 2;
            const zScaling = (avgZ - minZ) / (maxZ - minZ)
            if (lastMousePosition.x >= minCoordinate.x &&
                lastMousePosition.x <= maxCoordinate.x &&
                lastMousePosition.y >= minCoordinate.y &&
                lastMousePosition.y <= maxCoordinate.y) {
                targetAreaId = parseInt(overlayLabelsRows[o].otherColumnValues[1])
                targetPlaceName = overlayLabelsRows[o].otherColumnValues[0]
                targetX = avgX
                targetY = avgY
                targetFontSize = (((zScaling * 20 + 30)/2)*fontScale)
                connectionAreaIds = overlayRow.otherColumnValues[8].split(';').map(s => parseInt(s))
                cacheTargetCtx.fillStyle = "rgba(0, 0, 0, 0.9)";
                cacheTargetCtx.fillRect(minCoordinate.getCanvasX(), minCoordinate.getCanvasY(),
                    maxCoordinate.getCanvasX() - minCoordinate.getCanvasX(),
                    maxCoordinate.getCanvasY() - minCoordinate.getCanvasY())
            }
            if (drawOutlines) {
                cacheGridCtx.lineWidth = 0.25
                cacheGridCtx.strokeStyle = "black";
                cacheGridCtx.strokeRect(minCoordinate.getCanvasX(), minCoordinate.getCanvasY(),
                    maxCoordinate.getCanvasX() - minCoordinate.getCanvasX(),
                    maxCoordinate.getCanvasY() - minCoordinate.getCanvasY())
            }

        }
        mainCtx.drawImage(cacheGridCanvas, 0, 0);
        // draw colored fill ins for connections for connections
        for (let o = 0; drawTarget && o < overlayRows.length; o++) {
            const overlayRow = overlayRows[o]
            if (!connectionAreaIds.includes(parseInt(overlayRow.otherColumnValues[1]))) {
                continue
            }
            const minCoordinate = new MapCoordinate(
                parseFloat(overlayRow.otherColumnValues[2]),
                parseFloat(overlayRow.otherColumnValues[3]),
                false);
            const maxCoordinate = new MapCoordinate(
                parseFloat(overlayRow.otherColumnValues[5]),
                parseFloat(overlayRow.otherColumnValues[6]),
                false);
            const avgX = (minCoordinate.getCanvasX() + maxCoordinate.getCanvasX()) / 2
            const avgY = (minCoordinate.getCanvasY() + maxCoordinate.getCanvasY()) / 2
            const avgZ = (parseFloat(overlayRow.otherColumnValues[4]) + parseFloat(overlayRow.otherColumnValues[7])) / 2;
            const zScaling = (avgZ - minZ) / (maxZ - minZ)
            cacheTargetCtx.font = (((zScaling * 20 + 30)/2)*fontScale).toString() + "px Tahoma"
            cacheTargetCtx.fillStyle = "rgba(255, 0, 0, 0.2)";
            cacheTargetCtx.fillRect(minCoordinate.getCanvasX(), minCoordinate.getCanvasY(),
                maxCoordinate.getCanvasX() - minCoordinate.getCanvasX(),
                maxCoordinate.getCanvasY() - minCoordinate.getCanvasY())
        }
        if (drawTarget && targetAreaId != -1) {
            cacheTargetCtx.fillStyle = 'green'
            cacheTargetCtx.font = targetFontSize.toString() + "px Tahoma"
            cacheTargetCtx.fillText(targetAreaId.toString() + "," + targetPlaceName, targetX, targetY - 5.)
        }
        mainCtx.drawImage(cacheTargetCanvas, 0, 0);
    }
    else if (curOverlay.includes("reachable") || curOverlay.includes("visible") || curOverlay.includes("distance") ||
             curOverlay.includes("danger")) {
        mainCtx.fillStyle = green
        const overlayRows = filteredData.overlays.get(curOverlay)
        const overlayLabelsRows = filteredData.overlays.get(filteredData.parsers.get(curOverlay).overlayLabelsQuery)
        let distances: number[] = [];
        let minDistance;
        let maxDistance;
        let targetAreaId = -1
        let targetPlaceName = ""
        let targetX = -1
        let targetY = -1
        let targetFontSize = -1
        // if not already cached
        // draw all area outlines and compute target area
        let drawOutlines = false
        if (lastCacheOverlay == null || lastCacheOverlay != curOverlay) {
            lastCacheOverlay = curOverlay
            drawOutlines = true
            cacheGridCtx.clearRect(0, 0, cacheGridCanvas.width, cacheGridCanvas.height)
        }
        for (let o = 0; o < overlayRows.length; o++) {
            const overlayRow = overlayRows[o]
            const minCoordinate = new MapCoordinate(
                parseFloat(overlayRow.otherColumnValues[0]),
                parseFloat(overlayRow.otherColumnValues[1]),
                false);
            const maxCoordinate = new MapCoordinate(
                parseFloat(overlayRow.otherColumnValues[3]),
                parseFloat(overlayRow.otherColumnValues[4]),
                false);
            const avgX = (minCoordinate.getCanvasX() + maxCoordinate.getCanvasX()) / 2
            const avgY = (minCoordinate.getCanvasY() + maxCoordinate.getCanvasY()) / 2
            const avgZ = (parseFloat(overlayRow.otherColumnValues[2]) + parseFloat(overlayRow.otherColumnValues[5])) / 2;
            const zScaling = (avgZ - minZ) / (maxZ - minZ)
            if (lastMousePosition.x >= minCoordinate.x &&
                lastMousePosition.x <= maxCoordinate.x &&
                lastMousePosition.y >= minCoordinate.y &&
                lastMousePosition.y <= maxCoordinate.y) {
                targetAreaId = parseInt(overlayLabelsRows[o].otherColumnValues[1])
                targetPlaceName = overlayLabelsRows[o].otherColumnValues[0]
                targetX = avgX
                targetY = avgY
                targetFontSize = (((zScaling * 20 + 30)/2)*fontScale)
                distances = overlayRow.otherColumnValues.slice(6).map(s => parseFloat(s))
                minDistance = Math.min(...distances);
                maxDistance = Math.max(...distances);
                mainCtx.fillStyle = "rgba(0, 0, 0, 0.9)";
                mainCtx.fillRect(minCoordinate.getCanvasX(), minCoordinate.getCanvasY(),
                    maxCoordinate.getCanvasX() - minCoordinate.getCanvasX(),
                    maxCoordinate.getCanvasY() - minCoordinate.getCanvasY())
            }
            if (drawOutlines) {
                cacheGridCtx.lineWidth = 0.5
                cacheGridCtx.strokeStyle = "black";
                cacheGridCtx.strokeRect(minCoordinate.getCanvasX(), minCoordinate.getCanvasY(),
                    maxCoordinate.getCanvasX() - minCoordinate.getCanvasX(),
                    maxCoordinate.getCanvasY() - minCoordinate.getCanvasY())
            }
        }
        mainCtx.drawImage(cacheGridCanvas, 0, 0);
        // draw fill ins for all areas
        for (let o = 0; targetAreaId != -1 && o < overlayRows.length; o++) {
            const overlayRow = overlayRows[o]
            if (distances[o] == -1) {
                continue
            }
            const minCoordinate = new MapCoordinate(
                parseFloat(overlayRow.otherColumnValues[0]),
                parseFloat(overlayRow.otherColumnValues[1]),
                false);
            const maxCoordinate = new MapCoordinate(
                parseFloat(overlayRow.otherColumnValues[3]),
                parseFloat(overlayRow.otherColumnValues[4]),
                false);
            const percentDistance = (distances[o] - minDistance) / (maxDistance - minDistance);
            mainCtx.fillStyle = `rgba(${percentDistance * 255}, 0, ${(1 - percentDistance) * 255}, 0.5)`;
            mainCtx.fillRect(minCoordinate.getCanvasX(), minCoordinate.getCanvasY(),
                maxCoordinate.getCanvasX() - minCoordinate.getCanvasX(),
                maxCoordinate.getCanvasY() - minCoordinate.getCanvasY())
        }
        if (targetAreaId != -1) {
            mainCtx.fillStyle = 'green'
            mainCtx.font = targetFontSize.toString() + "px Tahoma"
            mainCtx.fillText(targetAreaId.toString() + "," + targetPlaceName, targetX, targetY)
        }
    }
    else if (curOverlay.includes("trajectory")) {
        mainCtx.fillStyle = green
        const overlayRows = filteredData.overlays.get(curOverlay)
        let curTrajectoryId = -1
        // if not already cached
        // draw all lines
        let drawLines = false
        if (lastCacheOverlay == null || lastCacheOverlay != curOverlay) {
            lastCacheOverlay = curOverlay
            drawLines = true
            cacheGridCtx.clearRect(0, 0, cacheGridCanvas.width, cacheGridCanvas.height)
        }
        for (let o = 0; o < overlayRows.length; o++) {
            const overlayRow = overlayRows[o]
            const fstCoordinate = new MapCoordinate(
                parseFloat(overlayRow.otherColumnValues[1]),
                parseFloat(overlayRow.otherColumnValues[2]),
                false);
            const sndCoordinate = new MapCoordinate(
                parseFloat(overlayRow.otherColumnValues[4]),
                parseFloat(overlayRow.otherColumnValues[5]),
                false);
            const avgX = (fstCoordinate.getCanvasX() + sndCoordinate.getCanvasX()) / 2
            const avgY = (fstCoordinate.getCanvasY() + sndCoordinate.getCanvasY()) / 2
            const avgZ = (parseFloat(overlayRow.otherColumnValues[3]) + parseFloat(overlayRow.otherColumnValues[6])) / 2;
            const zScaling = (avgZ - minZ) / (maxZ - minZ)
            if (drawLines) {
                cacheGridCtx.lineWidth = 0.5
                cacheGridCtx.strokeStyle = 'rgba(24,255,0,0.2)'
                if (overlayRow.foreignKeyValues[0] != curTrajectoryId) {
                    curTrajectoryId = overlayRow.foreignKeyValues[0]
                    cacheGridCtx.stroke()
                    cacheGridCtx.beginPath()
                    cacheGridCtx.moveTo(fstCoordinate.getCanvasX(), fstCoordinate.getCanvasY())
                }
                cacheGridCtx.lineTo(sndCoordinate.getCanvasX(), sndCoordinate.getCanvasY())
            }
        }
        // close last line
        if (drawLines) {
            cacheGridCtx.stroke()
        }
        mainCtx.drawImage(cacheGridCanvas, 0, 0);
    }
    setEventText(tickData, filteredData)
    // setup client config for this tick
    let startDemoTick = tickData.gameTickNumber - 100
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
    const numTicks = gameData.ticksTable.length
    for (let t = 0; t < numTicks; t++) {
        const tickData = gameData.ticksTable[t]
        const players = gameData.getPlayersAtTick(tickData)
        const round = gameData.getRound(tickData)
        for (let p = 0; p < players.length; p++) {
            minZ = Math.min(minZ, players[p].posZ)
            maxZ = Math.max(maxZ, players[p].posZ)
            if (!round.warmup) {
                maxViewY = Math.max(maxViewY, players[p].viewY)
                maxViewYT = t
                maxViewYP = p
            }
        }
    }
    drawTick(null)
}

export function setupCanvas() {
    mainCanvas = <HTMLCanvasElement> document.querySelector("#mainCanvas")
    mainCtx = mainCanvas.getContext('2d')
    cacheGridCanvas = <HTMLCanvasElement> document.createElement("canvas")
    cacheGridCtx = cacheGridCanvas.getContext('2d')
    cacheGridCanvas.width = mainCanvas.width
    cacheGridCanvas.height = mainCanvas.width
    cacheTargetCanvas = <HTMLCanvasElement> document.createElement("canvas")
    cacheTargetCtx = cacheTargetCanvas.getContext('2d')
    cacheTargetCanvas.width = mainCanvas.width
    cacheTargetCanvas.height = mainCanvas.width
    kymographCanvas = <HTMLCanvasElement> document.querySelector("#kymographCanvas")
    kymographCtx = kymographCanvas.getContext('2d')
    scatterCanvas = <HTMLCanvasElement> document.querySelector("#scatterCanvas")
    scatterCtx = scatterCanvas.getContext('2d')
    inferenceCanvas = <HTMLCanvasElement> document.querySelector("#inferenceCanvas")
    inferenceCtx = inferenceCanvas.getContext('2d')
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
    toggleCanvasSizeButton = document.querySelector<HTMLButtonElement>("#canvas_size")
    toggleCanvasSizeButton.addEventListener("click", toggleCanvasSize)
    mainCanvas.addEventListener("mousemove", trackMouse)
    mainCanvas.addEventListener("mousedown", startingRegionFilter)
    mainCanvas.addEventListener("mouseup", finishedRegionFilter)
    document.addEventListener('keydown', trackKeyDown)
    document.addEventListener('keyup', trackKeyUp)
    setupEventDrawing()
    createCharts(kymographCtx, scatterCtx, inferenceCtx)
}

function setEventsOverlaysAndRedraw() {
    setEventsOverlaysToDraw()
    drawTick(null)
}

export function setupCanvasHandlers() {
    document.querySelector<HTMLInputElement>("#tick-selector").addEventListener("input", drawTick)
    //document.querySelector<HTMLButtonElement>("#clear_filter").addEventListener("click", clearFilterButton)
    document.querySelector<HTMLSelectElement>("#event-type").addEventListener("change", setEventsOverlaysAndRedraw)
    document.querySelector<HTMLSelectElement>("#event-id-selector").addEventListener("change", setEventsOverlaysAndRedraw)
    document.querySelector<HTMLSelectElement>("#overlay-type").addEventListener("change", setEventsOverlaysAndRedraw)
    document.querySelector<HTMLSelectElement>("#clear_filter").addEventListener("click", clearFilterButton)
}

export function setupSmallOrLargeMode() {
    if (smallMode) {
        document.getElementById("button_rwd").style.display = "none"
        document.getElementById("button_play").style.display = "none"
        document.getElementById("button_ff").style.display = "none"
        document.getElementById("event-type-label").style.display = "none"
        document.getElementById("event-type").style.display = "none"
        document.getElementById("large-controls").style.display = "none"
        document.getElementById("canvas-data").style.display = "none"
        document.getElementById("canvas-data").style.display = "none"
        document.getElementById("copy_text").style.display = "none"
        document.getElementById("copy_button").style.display = "none"
        document.getElementById("match-info").style.display = "none"
        document.getElementById("round-info").style.display = "none"
    }
}