import {GameData, Row, TickRow} from "../data/tables";
import {curEvent} from "./events";

export function drawMouseData(kymographCanvas: HTMLCanvasElement, kymographCtx: CanvasRenderingContext2D,
                              gridCanvas: HTMLCanvasElement, gridCtx: CanvasRenderingContext2D,
                              gameData: GameData, tickData: TickRow, eventData: Row) {
    const parser = gameData.parsers.get(curEvent)
    if (parser.havePerTickAimTable) {
        kymographCanvas.style.display = "inline-block"
        gridCanvas.style.display = "inline-block"
        const aimDataPerTick = gameData.eventToPerTickAimTablesIndices.get(curEvent)
        
    }
    else {
        kymographCanvas.style.display = "none"
        gridCanvas.style.display = "none"
    }
}
