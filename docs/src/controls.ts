import { tickSelector, tickLabel, drawTick } from "./drawing"

let play1xButton: HTMLButtonElement = null;
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
    }
}

function stopPlaying() {
    if (curInterval !== null) {
        window.clearInterval(curInterval)
        curInterval = null
    }
}

let isPlaying = false
function playXSpeed(speed: number, playButton: boolean = false) {
    const img = <HTMLImageElement> play1xButton.children.item(0)
    stopPlaying()
    if (playButton && isPlaying) {
        playSpeed = 0
        img.src = "images/play-fill.svg"
        play1xButton.className = "btn btn-secondary playButton"
        rwdImg.className = "btn btn-secondary playButton"
        ffImg.className = "btn btn-secondary playButton"
        isPlaying = false
        ticksPerUpdate = 1
    }
    else {
        playSpeed = speed
        img.src = "images/pause-fill.svg"
        curInterval = window.setInterval(incrementTick, Math.max(1, 1000 / Math.abs(playSpeed)))
        play1xButton.className = "btn btn-green playButton"
        isPlaying = true
    }
}

function playFaster() {
    rwdImg.className = "btn btn-secondary playButton"
    if (playSpeed <= 0) {
        playXSpeed(speed1x * 2)
    }
    else if (playSpeed < 8 * speed1x) {
        playXSpeed(playSpeed * 2)
    }
    else if (playSpeed == 8 * speed1x) {
        ticksPerUpdate = 10
        playXSpeed(playSpeed)
    }
    if (playSpeed == speed1x * 2) {
        ffImg.className = "btn btn-blue playButton"
    }
    else if (playSpeed == speed1x * 4) {
        ffImg.className = "btn btn-green playButton"
    }
    else if (playSpeed == speed1x * 8 && ticksPerUpdate == 1) {
        ffImg.className = "btn btn-yellow playButton"
    }
    else if (playSpeed == speed1x * 8 && ticksPerUpdate == 10) {
        ffImg.className = "btn btn-red playButton"
    }
}

function playSlower() {
    ffImg.className = "btn btn-secondary playButton"
    if (playSpeed >= 0) {
        ticksPerUpdate = 1
        playXSpeed(speed1x * -1)
    }
    else if (playSpeed > -4 * speed1x) {
        playXSpeed(playSpeed * 2)
    }
    else if (playSpeed == -4 * speed1x) {
        ticksPerUpdate = 10
        playXSpeed(playSpeed * 2)
    }
    if (playSpeed == speed1x * -1) {
        rwdImg.className = "btn btn-blue playButton"
    }
    else if (playSpeed == speed1x * -2) {
        rwdImg.className = "btn btn-green playButton"
    }
    else if (playSpeed == speed1x * -4) {
        rwdImg.className = "btn btn-yellow playButton"
    }
    else if (playSpeed == speed1x * -8) {
        rwdImg.className = "btn btn-red playButton"
    }
}

export function registerPlayHandlers() {
    play1xButton = document.querySelector<HTMLButtonElement>("#button_play")
    ffImg = document.querySelector<HTMLImageElement>("#ff_img")
    rwdImg = document.querySelector<HTMLImageElement>("#rwd_img")
    document.querySelector<HTMLButtonElement>("#button_play")
        .addEventListener("click", () => playXSpeed(speed1x, true))
    document.querySelector<HTMLButtonElement>("#button_ff")
        .addEventListener("click", playFaster)
    document.querySelector<HTMLButtonElement>("#button_rwd")
        .addEventListener("click", playSlower)
    console.log(play1xButton)
}
