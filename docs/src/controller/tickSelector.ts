import {GameData} from "../data/tables";

export let tickSelector: HTMLInputElement = null;
export let tickLabel: HTMLLabelElement = null;
export let gameTickLabel: HTMLLabelElement = null;

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

export function setupTickSelector() {
    tickSelector = document.querySelector<HTMLInputElement>("#tick-selector")
    tickLabel = document.querySelector<HTMLLabelElement>("#cur-tick")
    tickLabel.innerHTML = "0"
    gameTickLabel = document.querySelector<HTMLLabelElement>("#game-tick")
}