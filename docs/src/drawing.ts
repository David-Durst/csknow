import {gameData, PositionRow} from "./data"

export const d2_top_left_x = -2476
export const d2_top_left_y = 3239
export const canvasWidth = 700
export const canvasHeight = 700
export const minimapWidth = 1024
export const minimapHeight = 1024
export const minimapScale = 4.4
export let canvas: HTMLCanvasElement = null;
export let ctx: CanvasRenderingContext2D = null;
let xMapLabel: HTMLLabelElement = null;
let yMapLabel: HTMLLabelElement = null;
let xCanvasLabel: HTMLLabelElement = null;
let yCanvasLabel: HTMLLabelElement = null;
let tickSelector: HTMLInputElement = null;
let tickLabel: HTMLLabelElement = null;
let tScoreLabel: HTMLLabelElement = null;
let ctScoreLabel: HTMLLabelElement = null;
let minZ = 0;
let maxZ = 0;

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

function trackMouse(e: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    const xCanvas = (e.clientX - rect.left)
    const yCanvas = (e.clientY - rect.top)
    const minimapCoordinate = new MapCoordinate(xCanvas, yCanvas, true)
    xMapLabel.innerHTML = minimapCoordinate.x.toPrecision(6)
    yMapLabel.innerHTML = minimapCoordinate.y.toPrecision(6)
    xCanvasLabel.innerHTML = minimapCoordinate.getCanvasX().toPrecision(6)
    yCanvasLabel.innerHTML = minimapCoordinate.getCanvasY().toPrecision(6)
}

export function drawTick(e: InputEvent) {
    const curTick = parseInt(tickSelector.value)
    tickLabel.innerHTML = curTick.toString()
    ctx.font = "30px Arial"
    tickSelector.max = gameData.position.length.toString()
    const tickData: PositionRow = gameData.position[curTick]
    tScoreLabel.innerHTML = tickData.tScore.toString()
    tScoreLabel.innerHTML = tickData.ctScore.toString()
    for (let p = 0; p < 10; p++) {
        let playerText = "o"
        if (!tickData.players[p].isAlive) {
            playerText = "x"
        }
        ctx.fillStyle = dark_blue
        if (tickData.players[p].team == 2) {
            ctx.fillStyle = dark_red
        }
        const location = new MapCoordinate(
            tickData.players[p].xPosition,
            tickData.players[p].yPosition,
            false);
        ctx.fillText(playerText, location.getCanvasX(), location.getCanvasY())
    }
}

export function setupMatch() {
    drawTick(null)
    const numTicks = gameData.position.length
    for (let t = 0; t < numTicks; t++) {
        for (let p = 0; p < 10; p++) {
            minZ = Math.min(minZ, gameData.position[t].players[p].zPosition)
            maxZ = Math.max(maxZ, gameData.position[t].players[p].zPosition)
        }
    }
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
    canvas.addEventListener("mousemove", trackMouse)
    document.querySelector<HTMLInputElement>("#tick-selector").addEventListener("input", drawTick)
    tickLabel.innerHTML = "0"
}