import {getEventIndex, PositionRow} from "../data/tables";
import {drawTick} from "../drawing/drawing";
import {filteredData} from "./filter";
import {curEvent} from "../drawing/events";
import {setCurTickIndex, setTickSelectorMax } from "./tickSelector";

let customFilterEditor: HTMLDivElement = null
let customFilterText: HTMLDivElement = null
let filterUploadButton: HTMLInputElement = null


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
    setTickSelectorMax(filteredData.position.length - 1)
    setCurTickIndex(0);
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

function updateCustomFilterProgram(e: any) {
    filterUploadButton.disabled = true
    e.currentTarget.files[0].text().then((text: string) => {
        // @ts-ignore
        window.editor.setValue(text)
        filterUploadButton.disabled = false
    })
}

function saveCustomFilterProgram() {
    // @ts-ignore
    const blob = new Blob([window.editor.getValue()],
        {type: "text/javascript;charset=utf-8"});
    //FileSaver.
}

export function setupCustomFilters() {
    customFilterEditor = document.querySelector<HTMLDivElement>("#editor")
    document.querySelector<HTMLButtonElement>("#custom_filter").addEventListener("click", customFilterButton)
    customFilterText = document.querySelector<HTMLDivElement>("#custom_filter_text")
    filterUploadButton = document.querySelector<HTMLInputElement>("#upload_custom_filter")
    document.querySelector<HTMLInputElement>("#upload_custom_filter").addEventListener("change", updateCustomFilterProgram)
}
