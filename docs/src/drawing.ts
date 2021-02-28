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
}
let yLabel: HTMLLabelElement = null;
export function setYLabel(inputYLabel: HTMLLabelElement) {
    yLabel = inputYLabel
}

function trackMouse(e: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    xLabel.innerHTML = (e.clientX - rect.left).toString()
    yLabel.innerHTML = (e.clientY - rect.top).toString()
}

//canvas.addEventListener("onmousemove", trackMouse)