import {getEventIndex, PositionRow} from "../data/tables";
import {drawTick} from "../drawing/drawing";
import {filteredData} from "./filter";
import {curEvent} from "../drawing/events";

let customFilterEditor: HTMLDivElement = null
let customFilterText: HTMLDivElement = null


let shouldFilterCustom: boolean = false
export function customFilter() {
    let localData = filteredData
    customFilterText.innerHTML = "starting custom filter"
    if (!shouldFilterCustom) {
        return true;
    }
    let matchingPositions: PositionRow[] = []
    // @ts-ignore
    eval(window.editor.getValue())
    console.log("ran, matching positions size: " + matchingPositions.length.toString())
    if (matchingPositions.length == 0) {
        customFilterText.innerHTML = "found no values"
        return false;
    }
    filteredData.position = matchingPositions
    customFilterText.innerHTML = "successfully applied filter"
    drawTick(null)
    return true;
}

function customFilterButton() {
    shouldFilterCustom = true
    customFilter()
}

export function clearCustomFilter() {
    customFilterText.innerHTML = "not applied"
    shouldFilterCustom = false
}

export function setupCustomFilters() {
    customFilterEditor = document.querySelector<HTMLDivElement>("#editor")
    document.querySelector<HTMLButtonElement>("#custom_filter").addEventListener("click", customFilterButton)
    customFilterText = document.querySelector<HTMLDivElement>("#custom_filter_text")
}
