import {GameData} from "../data/tables";
import {gameData} from "../data/data";

export let tickSelector: HTMLInputElement = null;
export let tickLabel: HTMLLabelElement = null;
export let gameTickLabel: HTMLLabelElement = null;
export let matchSelector: HTMLInputElement = null;
let matchLabel: HTMLLabelElement = null;
export let roundSelector: HTMLInputElement = null;
let roundLabel: HTMLLabelElement = null;

export function setTickSelectorMax(value: number) {
    tickSelector.max = value.toString()
}

export function getCurTickIndex(): number {
    return parseInt(tickSelector.value)
}

export function setCurTickIndex(value: number) {
    tickSelector.value = value.toString()
}

export function setTickLabel(value: number, gameTick: number) {
    tickLabel.innerHTML = value.toString()
    gameTickLabel.innerHTML = gameTick.toString()
}

export function setupSelectors(gameData: GameData) {
    matchLabel = document.querySelector<HTMLLabelElement>("#cur-match")
    matchSelector = document.querySelector<HTMLInputElement>("#match-selector")
    matchSelector.value = "0"
    matchSelector.min = "0"
    matchSelector.max = gameData.gamesTable.length.toString()

    roundLabel = document.querySelector<HTMLLabelElement>("#cur-round")
    roundSelector = document.querySelector<HTMLInputElement>("#round-selector")
    roundSelector.value = "0"
    roundSelector.min = "0"
    roundSelector.max = gameData.roundsTable.length.toString()

    tickSelector = document.querySelector<HTMLInputElement>("#tick-selector")
    tickLabel = document.querySelector<HTMLLabelElement>("#cur-tick")
    tickSelector.value = "0"
    tickSelector.min = "0"
    tickSelector.max = gameData.ticksTable.length.toString()
    gameTickLabel = document.querySelector<HTMLLabelElement>("#game-tick")
}