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

let xLabel: HTMLLabelElement = null;
export function setXLabel(inputXLabel: HTMLLabelElement) {
    xLabel = inputXLabel
    console.log("set x label to ")
    console.log(xLabel)
}
let yLabel: HTMLLabelElement = null;
export function setYLabel(inputYLabel: HTMLLabelElement) {
    yLabel = inputYLabel
}

// see last post by randunel and csgo/resources/overview/de_dust2.txt
// https://forums.alliedmods.net/showthread.php?p=2690857#post2690857
class MinimapCoordinate {
    x: number
    y: number

    constructor(x: number, y: number, fromCanvasPixels: boolean) {
        if (fromCanvasPixels) {
            const pctX = x / 700;
            this.x = d2_top_left_x + minimapScale * minimapWidth * pctX
            const pctY = y / 700;
            this.y = d2_top_left_y - minimapScale * minimapHeight * pctY
        } else {
            this.x = x
            this.y = y
        }
    }
}

function trackMouse(e: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    const xCanvas = (e.clientX - rect.left)
    const yCanvas = (e.clientY - rect.top)
    const minimapCoordinate = new MinimapCoordinate(xCanvas, yCanvas, true)
    xLabel.innerHTML = minimapCoordinate.x.toPrecision(6)
    yLabel.innerHTML = minimapCoordinate.y.toPrecision(6)
}

export function setup() {
    canvas.addEventListener("mousemove", trackMouse)
}