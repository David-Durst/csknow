import { tickSelector, tickLabel, drawTick } from "./drawing"

let play1xButton: HTMLButtonElement = null;
const playSpeed: number = 0;

function registerPlayHandlers() {
    play1xButton = document.querySelector<HTMLButtonElement>("#button_play")
    console.log(play1xButton)
}
