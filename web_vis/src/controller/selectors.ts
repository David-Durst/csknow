import {GameData} from "../data/tables";
import {gameData} from "../data/data";

export let tickSelector: HTMLInputElement = null;
export let tickLabel: HTMLLabelElement = null;
export let gameTickLabel: HTMLLabelElement = null;
export let matchSelector: HTMLInputElement = null;
export let matchLabel: HTMLLabelElement = null;
export let roundSelector: HTMLInputElement = null;
export let roundLabel: HTMLLabelElement = null;

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

export function setRoundsSelectorMax(value: number) {
    roundSelector.max = value.toString()
}

export function setupSelectors(gameData: GameData) {
    matchLabel = document.querySelector<HTMLLabelElement>("#cur-match")
    matchSelector = document.querySelector<HTMLInputElement>("#match-selector")
    matchSelector.value = "0"
    matchSelector.min = "0"
    const matchLength: number = gameData.gamesTable.length - 1
    matchSelector.max = matchLength.toString()

    roundLabel = document.querySelector<HTMLLabelElement>("#cur-round")
    roundSelector = document.querySelector<HTMLInputElement>("#round-selector")
    roundSelector.value = "0"
    roundSelector.min = "0"
    const roundsLength: number = gameData.roundsTable.length - 1
    roundSelector.max = roundsLength.toString()

    tickSelector = document.querySelector<HTMLInputElement>("#tick-selector")
    tickLabel = document.querySelector<HTMLLabelElement>("#cur-tick")
    tickSelector.value = "0"
    tickSelector.min = "0"
    const ticksLength: number = gameData.ticksTable.length - 1
    tickSelector.max = ticksLength.toString()
    gameTickLabel = document.querySelector<HTMLLabelElement>("#game-tick")
}