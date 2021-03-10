export let tickSelector: HTMLInputElement = null;
export let tickLabel: HTMLLabelElement = null;

export function setTickSelectorMax(value: number) {
    tickSelector.max = value.toString()
}

export function getCurTickIndex(): number {
    return parseInt(tickSelector.value)
}

export function setCurTickIndex(value: number) {
    tickSelector.value = value.toString()
}

export function setTickLabel(value: number) {
    tickLabel.innerHTML = value.toString()
}

export function setupTickSelector() {
    tickSelector = document.querySelector<HTMLInputElement>("#tick-selector")
    tickLabel = document.querySelector<HTMLLabelElement>("#cur-tick")
    tickLabel.innerHTML = "0"
}