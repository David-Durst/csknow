export const d2_top_left_x = -2476
export const d2_top_left_y = 3239
export const canvasWidth = 700
export const canvasHeight = 700
export const minimapWidth = 1024
export const minimapHeight = 1024
export const minimapScale = 4.4
export let canvas: HTMLCanvasElement = null;
export function setCanvas(inputCanvas: HTMLCanvasElement) {
    canvas = inputCanvas
}
export let ctx: CanvasRenderingContext2D = null;
export function setCTX(inputCTX: CanvasRenderingContext2D) {
    ctx = inputCTX
}

let xMapLabel: HTMLLabelElement = null;
export function setXMapLabel(inputXLabel: HTMLLabelElement) {
    xMapLabel = inputXLabel
}
let yMapLabel: HTMLLabelElement = null;
export function setYMapLabel(inputYLabel: HTMLLabelElement) {
    yMapLabel = inputYLabel
}

let xCanvasLabel: HTMLLabelElement = null;
export function setXCanvasLabel(inputXLabel: HTMLLabelElement) {
    xCanvasLabel = inputXLabel
}
let yCanvasLabel: HTMLLabelElement = null;
export function setYCanvasLabel(inputYLabel: HTMLLabelElement) {
    yCanvasLabel = inputYLabel
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

export function setup() {
    canvas.addEventListener("mousemove", trackMouse)
}