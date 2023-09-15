import {gameData, initialized} from "../data/data"
import {
    filteredData,
    applyJustEventFilter,
    filterRegion, stopFilteringEvents
} from "../controller/filter";
import {parseBool, PlayerAtTickRow, PlayerRow, Row, TickRow} from "../data/tables";
import {getPackedSettings} from "http2";
import {
    activeEvent,
    curOverlay, DEFAULT_ALIVE_STRING,
    getPlayersText, getPosTextPositions, setEventsOverlaysToDraw,
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
let togglePlayersButton: HTMLButtonElement = null;
let showAllNavsButton: HTMLButtonElement = null;
let showPlayers = true;
let showAllNavs = false;
let minZ = 0;
let maxZ = 0;
const black = "rgba(0,0,0,1.0)";
const white_translucent = "rgba(255,255,255,0.6)";
const gray = "rgba(159,159,159,1.0)";
const dark_blue = "rgba(4,190,196,1.0)";
const light_blue = "rgba(194,255,243,1.0)";
const purple = "rgb(160,124,205)";
const dark_red = "rgba(209,0,0,1.0)";
const light_red = "rgba(255,143,143,1.0)";
const yellow = "rgb(187,142,52)";
const green = "rgba(0,150,0,1.0)";
export let smallMode: boolean = false
const smoke_gray = "rgba(200, 200, 200, 0.4)"
const smokeGrenadeImg = new Image()
smokeGrenadeImg.src = "vis_images/smoke_grenade.png"
function getFlashColor(alpha: number): string {
    return `rgba(255, 255, 255, ${alpha})`
}

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
    lastCacheOverlay = null
    drawTick(null)
}

export function togglePlayers() {
    if (showPlayers) {
        togglePlayersButton.innerText = "show players"
    }
    else {
        togglePlayersButton.innerText = "hide players"
    }
    showPlayers = !showPlayers
    drawTick(null)
}

export function toggleAllNavs() {
    if (showAllNavs) {
        showAllNavsButton.innerText = "show all navs"
    }
    else {
        showAllNavsButton.innerText = "show selected navs"
    }
    showAllNavs = !showAllNavs
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

let layerToDraw: number = 9
let altPressed: boolean = false
let controlPressed: boolean = false
let shiftPressed: boolean = false
let activeKeys: Set<string> = new Set<string>()
function trackKeyDown(e: KeyboardEvent) {
    activeKeys.add(e.key)
    altPressed = e.altKey
    controlPressed = e.ctrlKey
    shiftPressed = e.shiftKey
    if (!Number.isNaN(parseInt(e.key))) {
        layerToDraw = parseInt(e.key)
    }
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

class TargetAreaData {
    index: number;
    avgX: number;
    avgY: number;
    avgZ: number;
    overlayRow: Row;
    overlayLabelsRow: Row;
    minCoordinate: MapCoordinate;
    maxCoordinate: MapCoordinate;

    constructor(index: number, avgX: number, avgY: number, avgZ: number, overlayRow: Row, overlayLabelsRow: Row,
                minCoordinate: MapCoordinate, maxCoordinate: MapCoordinate) {
        this.index = index
        this.avgX = avgX
        this.avgY = avgY
        this.avgZ = avgZ
        this.overlayRow = overlayRow
        this.overlayLabelsRow = overlayLabelsRow
        this.minCoordinate = minCoordinate
        this.maxCoordinate = maxCoordinate
    }
}

let priorFrameShowAllNavs = false
let removedAreas = new Set([
    6938, 9026, // these are barrels on A that I get stuck on
    8251, // this one is under t spawn
    8631, // this one is on cat next to boxes, weird
    //4232, 4417, // bad wall and box on long
    8531, // mid doors ct side
    8753, 8550, 8574, // b car
    8594, 8600, 8601, 8602, 8607, // boxes under cat to a
    8966, 8967, 8970, 8969, 8968, // t spawn
    //8973 // under hole inside B
    3973, 3999, 4000, // out of bounds near b tunnels entran
])

export function drawTick(e: InputEvent) {
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
    const playerFlashed = gameData.getPlayerFlashedThisTick(tickData)
    for (let p = 0; p < players.length && showPlayers; p++) {
        let playerText = playersText.get(players[p].playerId)
        mainCtx.fillStyle = dark_blue
        if (players[p].team == 3) {
            if (players[p].playerId == selectedPlayer) {
                mainCtx.fillStyle = purple
            }
            /*
            else if (playerText == "t" || playerText == "s") {
                mainCtx.fillStyle = light_blue
            }
             */
        }
        else {
            mainCtx.fillStyle = yellow
            if (players[p].playerId == selectedPlayer) {
                mainCtx.fillStyle = dark_red
            }
            /*
            else if (playerText == "t" || playerText == "s") {
                mainCtx.fillStyle = yellow
            }
             */
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
        if (playerFlashed.has(players[p].playerId)) {
            mainCtx.fillStyle = getFlashColor(playerFlashed.get(players[p].playerId));
            mainCtx.fillText(playerText, location.getCanvasX(), location.getCanvasY())
        }
        //ctx.fillRect(location.getCanvasX(), location.getCanvasY(), 1, 1)
    }
    const smokes = gameData.getSmokeGrenades(tickData)
    // https://www.google.com/search?client=firefox-b-1-d&q=smoke+radius+csgo
    for (let s = 0; s < smokes.length; s++) {
        const smokeLocation = new MapCoordinate(
            smokes[s].posX,
            smokes[s].posY,
            false);
        const smokeMax = new MapCoordinate(
            smokes[s].posX + 144,
            smokes[s].posY + 144,
            false);
        const smokeRadius = smokeMax.getCanvasX() - smokeLocation.getCanvasX()
        if (smokes[s].state == 0) {
            mainCtx.drawImage(smokeGrenadeImg, smokeLocation.getCanvasX(), smokeLocation.getCanvasY(), 30, 30)
        }
        else {
            mainCtx.beginPath();
            mainCtx.arc(smokeLocation.getCanvasX(), smokeLocation.getCanvasY(), smokeRadius,
                0, 2 * Math.PI, false);
            mainCtx.fillStyle = smoke_gray
            mainCtx.fill()
        }
    }
    const posTextPositions = getPosTextPositions(tickData, filteredData)
    mainCtx.fillStyle = white_translucent
    for (let ptp = 0; ptp < posTextPositions.length; ptp++) {
        const zScaling = (posTextPositions[ptp].pos.posZ - minZ) / (maxZ - minZ)
        if (posTextPositions[ptp].small) {
            mainCtx.font = (8 * fontScale).toString() + "px Arial"
        }
        else {
            mainCtx.font = ((zScaling * 5 + 14) * fontScale).toString() + "px Arial"
        }
        const location = new MapCoordinate(
            posTextPositions[ptp].pos.posX,
            posTextPositions[ptp].pos.posY,
            false);
        mainCtx.fillText(posTextPositions[ptp].text, location.getCanvasX(), location.getCanvasY())
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
    // if not already cached for the cur overlay
    // draw all area outlines and compute target area
    let drawOutlines = false
    if (lastCacheOverlay == null || lastCacheOverlay != curOverlay || showAllNavs || priorFrameShowAllNavs) {
        lastCacheOverlay = curOverlay
        drawOutlines = true
        cacheGridCtx.clearRect(0, 0, cacheGridCanvas.width, cacheGridCanvas.height)
    }
    // update target only when mouse inside canvas and target selector key pressed
    let drawTarget =
        lastMousePosition.getCanvasX() > 0. &&
        lastMousePosition.getCanvasX() < mainCanvas.width &&
        lastMousePosition.getCanvasY() > 0. &&
        lastMousePosition.getCanvasY() < mainCanvas.height &&
        controlPressed;
    if (drawOutlines || drawTarget || showAllNavs || priorFrameShowAllNavs) {
        cacheTargetCtx.clearRect(0, 0, cacheTargetCanvas.width, cacheTargetCanvas.height)
    }
    priorFrameShowAllNavs = showAllNavs
    if ((curOverlay.includes("mesh") || curOverlay.includes("cells")) && !curOverlay.includes("visible")) {
        mainCtx.fillStyle = green
        const overlayRows = filteredData.overlays.get(curOverlay)
        const overlayLabelsRows = filteredData.overlays.get(filteredData.parsers.get(curOverlay).overlayLabelsQuery)
        let connectionAreaIds: number[] = [];
        let possibleTargetAreas: TargetAreaData[] = []
        let targetAreaId = -1
        let targetId = -1
        let targetPlaceName = ""
        let targetX = -1
        let targetY = -1
        let targetFontSize = -1
        for (let o = 0; (drawOutlines || drawTarget) && o < overlayRows.length; o++) {
            const overlayRow = overlayRows[o]
            const overlayLabelsRow = overlayLabelsRows[o]
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
            if (lastMousePosition.x >= minCoordinate.x &&
                lastMousePosition.x <= maxCoordinate.x &&
                lastMousePosition.y >= minCoordinate.y &&
                lastMousePosition.y <= maxCoordinate.y) {
                possibleTargetAreas.push(new TargetAreaData(o, avgX, avgY, avgZ, overlayRow, overlayLabelsRow,
                    minCoordinate, maxCoordinate))
            }
            if (drawOutlines) {
                cacheGridCtx.lineWidth = 0.25
                cacheGridCtx.strokeStyle = "black";
                cacheGridCtx.strokeRect(minCoordinate.getCanvasX(), minCoordinate.getCanvasY(),
                    maxCoordinate.getCanvasX() - minCoordinate.getCanvasX(),
                    maxCoordinate.getCanvasY() - minCoordinate.getCanvasY())
            }
        }

        if (possibleTargetAreas.length > 0)
        {
            possibleTargetAreas.sort((a, b) => {return a.avgZ - b.avgZ});
            const possibleTargetArea = possibleTargetAreas[Math.min(possibleTargetAreas.length - 1, layerToDraw)]
            targetAreaId = parseInt(possibleTargetArea.overlayRow.otherColumnValues[1])
            targetId = possibleTargetArea.overlayRow.id
            targetPlaceName = possibleTargetArea.overlayRow.otherColumnValues[0]
            targetX = possibleTargetArea.avgX
            targetY = possibleTargetArea.avgY
            const zScaling = (possibleTargetArea.avgZ - minZ) / (maxZ - minZ)
            targetFontSize = (((zScaling * 20 + 30)/2)*fontScale)
            connectionAreaIds = possibleTargetArea.overlayRow.otherColumnValues[8].split(';').map(s => parseInt(s))
            cacheTargetCtx.fillStyle = "rgba(0,42,255,0.9)";
            cacheTargetCtx.fillRect(possibleTargetArea.minCoordinate.getCanvasX(), possibleTargetArea.minCoordinate.getCanvasY(),
                possibleTargetArea.maxCoordinate.getCanvasX() - possibleTargetArea.minCoordinate.getCanvasX(),
                possibleTargetArea.maxCoordinate.getCanvasY() - possibleTargetArea.minCoordinate.getCanvasY())
        }

        mainCtx.drawImage(cacheGridCanvas, 0, 0);
        // draw colored fill ins for connections for connections
        for (let o = 0; drawTarget && targetAreaId != -1 && o < overlayRows.length; o++) {
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
            cacheTargetCtx.fillText(targetId.toString() + "," + targetAreaId.toString() + "," + targetPlaceName, targetX, targetY - 5.)
        }
        mainCtx.drawImage(cacheTargetCanvas, 0, 0);
    }
    else if (curOverlay.includes("reachable") || curOverlay.includes("visible") || curOverlay.includes("distance") ||
             curOverlay.includes("danger")) {
        mainCtx.fillStyle = green
        const overlayRows = filteredData.overlays.get(curOverlay)
        const overlayLabelsRows = filteredData.overlays.get(filteredData.parsers.get(curOverlay).overlayLabelsQuery)
        const curParser = filteredData.parsers.get(curOverlay)
        let valuesForColor: number[] = [];
        let minValueForColor;
        let maxValueForColor;
        let possibleTargetAreas: TargetAreaData[] = []
        let targetAreaId = -1
        let targetAreaIndex = -1
        let targetPlaceName = ""
        let targetX = -1
        let targetY = -1
        let targetFontSize = -1
        for (let o = 0; !showAllNavs && (drawOutlines || drawTarget) && o < overlayLabelsRows.length; o++) {
            const overlayRow = overlayRows[o]
            const overlayLabelsRow = overlayLabelsRows[o]
            const minCoordinate = new MapCoordinate(
                parseFloat(overlayLabelsRow.otherColumnValues[2]),
                parseFloat(overlayLabelsRow.otherColumnValues[3]),
                false);
            const maxCoordinate = new MapCoordinate(
                parseFloat(overlayLabelsRow.otherColumnValues[5]),
                parseFloat(overlayLabelsRow.otherColumnValues[6]),
                false);
            const avgX = (minCoordinate.getCanvasX() + maxCoordinate.getCanvasX()) / 2
            const avgY = (minCoordinate.getCanvasY() + maxCoordinate.getCanvasY()) / 2
            const avgZ = (parseFloat(overlayLabelsRow.otherColumnValues[4]) + parseFloat(overlayLabelsRow.otherColumnValues[7])) / 2;
            if (lastMousePosition.x >= minCoordinate.x &&
                lastMousePosition.x <= maxCoordinate.x &&
                lastMousePosition.y >= minCoordinate.y &&
                lastMousePosition.y <= maxCoordinate.y) {
                possibleTargetAreas.push(new TargetAreaData(o, avgX, avgY, avgZ, overlayRow, overlayLabelsRow,
                    minCoordinate, maxCoordinate))
            }
            if (drawOutlines) {
                cacheGridCtx.lineWidth = 0.25
                cacheGridCtx.strokeStyle = "black";
                cacheGridCtx.strokeRect(minCoordinate.getCanvasX(), minCoordinate.getCanvasY(),
                    maxCoordinate.getCanvasX() - minCoordinate.getCanvasX(),
                    maxCoordinate.getCanvasY() - minCoordinate.getCanvasY())
            }
        }

        if (possibleTargetAreas.length > 0)
        {
            possibleTargetAreas.sort((a, b) => {return a.avgZ - b.avgZ});
            const possibleTargetArea = possibleTargetAreas[Math.min(possibleTargetAreas.length - 1, layerToDraw)]
            targetAreaId = parseInt(overlayLabelsRows[possibleTargetArea.index].otherColumnValues[1])
            targetAreaIndex = possibleTargetArea.index
            targetPlaceName = overlayLabelsRows[possibleTargetArea.index].otherColumnValues[0]
            targetX = possibleTargetArea.avgX
            targetY = possibleTargetArea.avgY
            const zScaling = (possibleTargetArea.avgZ - minZ) / (maxZ - minZ)
            targetFontSize = (((zScaling * 20 + 30)/2)*fontScale)
            if (!curOverlay.includes("visible")) {
                valuesForColor = possibleTargetArea.overlayRow.otherColumnValues.slice(6).map(s => parseFloat(s))
                minValueForColor = Math.min(...valuesForColor);
                maxValueForColor = Math.max(...valuesForColor);
            }
            cacheTargetCtx.fillStyle = "rgba(0, 0, 0, 0.9)";
            cacheTargetCtx.fillRect(possibleTargetArea.minCoordinate.getCanvasX(), possibleTargetArea.minCoordinate.getCanvasY(),
                possibleTargetArea.maxCoordinate.getCanvasX() - possibleTargetArea.minCoordinate.getCanvasX(),
                possibleTargetArea.maxCoordinate.getCanvasY() - possibleTargetArea.minCoordinate.getCanvasY())
        }

        mainCtx.drawImage(cacheGridCanvas, 0, 0);
        // draw fill ins for all areas
        for (let o = 0; (showAllNavs || (drawTarget && targetAreaId != -1)) && o < overlayLabelsRows.length; o++) {
            // this is area id
            if (showAllNavs && removedAreas.has(parseInt(overlayLabelsRows[o].otherColumnValues[1]))) {
                continue
            }
            if (curOverlay.includes("visible")) {
                const visDirA = curParser.blobAsMatrixValue(targetAreaIndex, o);
                const visDirB = curParser.blobAsMatrixValue(o, targetAreaIndex);
                if (visDirA || visDirB) {
                    cacheTargetCtx.fillStyle = `rgba(0, 0, 255, 0.5)`;
                    const overlayLabelsRow = overlayLabelsRows[o]
                    const minCoordinate = new MapCoordinate(
                        parseFloat(overlayLabelsRow.otherColumnValues[2]),
                        parseFloat(overlayLabelsRow.otherColumnValues[3]),
                        false);
                    const maxCoordinate = new MapCoordinate(
                        parseFloat(overlayLabelsRow.otherColumnValues[5]),
                        parseFloat(overlayLabelsRow.otherColumnValues[6]),
                        false);
                    cacheTargetCtx.fillRect(minCoordinate.getCanvasX(), minCoordinate.getCanvasY(),
                        maxCoordinate.getCanvasX() - minCoordinate.getCanvasX(),
                        maxCoordinate.getCanvasY() - minCoordinate.getCanvasY())
                }
            }
            else {
                const overlayLabelsRow = overlayLabelsRows[o]
                if (valuesForColor[o] == -1) {
                    continue
                }
                const minCoordinate = new MapCoordinate(
                    parseFloat(overlayLabelsRow.otherColumnValues[2]),
                    parseFloat(overlayLabelsRow.otherColumnValues[3]),
                    false);
                const maxCoordinate = new MapCoordinate(
                    parseFloat(overlayLabelsRow.otherColumnValues[5]),
                    parseFloat(overlayLabelsRow.otherColumnValues[6]),
                    false);
                if (showAllNavs) {
                    cacheTargetCtx.fillStyle = `rgba(0, 150, 0, 1.0)`;
                }
                else {
                    const percentDistance = (valuesForColor[o] - minValueForColor) / (maxValueForColor - minValueForColor);
                    cacheTargetCtx.fillStyle = `rgba(${percentDistance * 255}, 0, ${(1 - percentDistance) * 255}, 0.5)`;
                }
                cacheTargetCtx.fillRect(minCoordinate.getCanvasX(), minCoordinate.getCanvasY(),
                    maxCoordinate.getCanvasX() - minCoordinate.getCanvasX(),
                    maxCoordinate.getCanvasY() - minCoordinate.getCanvasY())
                if (showAllNavs) {
                    cacheTargetCtx.lineWidth = 1.
                    cacheTargetCtx.strokeStyle = "orange";
                    cacheTargetCtx.strokeRect(minCoordinate.getCanvasX(), minCoordinate.getCanvasY(),
                        maxCoordinate.getCanvasX() - minCoordinate.getCanvasX(),
                        maxCoordinate.getCanvasY() - minCoordinate.getCanvasY())
                }
            }
        }
        if (drawTarget && targetAreaId != -1) {
            cacheTargetCtx.fillStyle = 'green'
            cacheTargetCtx.font = targetFontSize.toString() + "px Tahoma"
            cacheTargetCtx.fillText(targetAreaId.toString() + "," + targetPlaceName, targetX, targetY - 5.)
        }

        mainCtx.drawImage(cacheTargetCanvas, 0, 0);
    }
    else if (curOverlay.includes("trajectory")) {
        mainCtx.fillStyle = green
        const overlayRows = filteredData.overlays.get(curOverlay)
        let curTrajectoryId = -1
        for (let o = 0; drawOutlines && o < overlayRows.length; o++) {
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
            if (drawOutlines) {
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
        if (drawOutlines) {
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
    togglePlayersButton = document.querySelector<HTMLButtonElement>("#players_toggle")
    togglePlayersButton.addEventListener("click", togglePlayers)
    showAllNavsButton = document.querySelector<HTMLButtonElement>("#show_all_navs")
    showAllNavsButton.addEventListener("click", toggleAllNavs)
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