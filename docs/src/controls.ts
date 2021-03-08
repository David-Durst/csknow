import { drawTick } from "./drawing/drawing"
import { tickSelector, tickLabel } from "./filter";

let playButton: HTMLButtonElement = null;
let ffButton: HTMLButtonElement = null;
let rwdButton: HTMLButtonElement = null;
let playImg: HTMLImageElement = null;
let ffImg: HTMLImageElement = null;
let rwdImg: HTMLImageElement = null;
let playSpeed: number = 0;
let curInterval: number = null;
const speed1x: number = 32
let ticksPerUpdate: number = 1

function incrementTick() {
    const tickIncrement = ticksPerUpdate * (playSpeed > 0 ? 1 : -1)
    const newTick = parseInt(tickSelector.value) + tickIncrement
    if (newTick <= parseInt(tickSelector.max) && newTick >= parseInt(tickSelector.min)) {
        tickSelector.value = newTick.toString()
        tickLabel.innerHTML = newTick.toString()
        drawTick(null)
    }
    else {
        stopPlaying()
        playButton.className = "btn btn-secondary playButton"
        rwdButton.className = "btn btn-secondary playButton"
        ffButton.className = "btn btn-secondary playButton"
    }
}

function stopPlaying() {
    if (curInterval !== null) {
        window.clearInterval(curInterval)
        curInterval = null
    }
}

let isPlaying = false
function playXSpeed(speed: number, isPlayButton: boolean = false) {
    stopPlaying()
    if (isPlayButton && isPlaying) {
        playSpeed = 0
        playImg.src = "images/play-fill.svg"
        playButton.className = "btn btn-secondary playButton"
        rwdButton.className = "btn btn-secondary playButton"
        ffButton.className = "btn btn-secondary playButton"
        isPlaying = false
        ticksPerUpdate = 1
    }
    else {
        playSpeed = speed
        playImg.src = "images/pause-fill.svg"
        curInterval = window.setInterval(incrementTick, Math.max(1, 1000 / Math.abs(playSpeed)))
        playButton.className = "btn btn-green playButton"
        isPlaying = true
    }
}

function playFaster() {
    rwdButton.className = "btn btn-secondary playButton"
    if (playSpeed <= speed1x) {
        playXSpeed(speed1x * 2)
    }
    else if (playSpeed == 2 * speed1x) {
        ticksPerUpdate = 10
        playXSpeed(speed1x * 8)
    }
    if (playSpeed == speed1x * 2) {
        ffButton.className = "btn btn-blue playButton"
    }
    else if (playSpeed == speed1x * 8 && ticksPerUpdate == 10) {
        ffButton.className = "btn btn-red playButton"
    }
}

function playSlower() {
    console.log("playSpeed: " + playSpeed.toString())
    ffButton.className = "btn btn-secondary playButton"
    if (playSpeed >= 0) {
        ticksPerUpdate = 1
        playXSpeed(speed1x * -2)
    }
    else if (playSpeed == -2 * speed1x) {
        ticksPerUpdate = 10
        playXSpeed(speed1x * -8)
    }
    if (playSpeed == speed1x * -2) {
        rwdButton.className = "btn btn-blue playButton"
    }
    else if (playSpeed == speed1x * -8) {
        rwdButton.className = "btn btn-red playButton"
    }
}

export function registerPlayHandlers() {
    playButton = document.querySelector<HTMLButtonElement>("#button_play")
    ffButton = document.querySelector<HTMLButtonElement>("#button_ff")
    rwdButton = document.querySelector<HTMLButtonElement>("#button_rwd")
    playImg = document.querySelector<HTMLImageElement>("#play_img")
    ffImg = document.querySelector<HTMLImageElement>("#ff_img")
    rwdImg = document.querySelector<HTMLImageElement>("#rwd_img")
    playButton.addEventListener("click", () => playXSpeed(speed1x, true))
    ffButton.addEventListener("click", playFaster)
    rwdButton.addEventListener("click", playSlower)
}
