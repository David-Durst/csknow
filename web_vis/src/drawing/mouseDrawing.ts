import {GameData, Parser, Row, TickRow} from "../data/tables";
import {curEvent} from "./events";
import {Chart, registerables, ScatterDataPoint} from 'chart.js';
Chart.register(...registerables);

let kymographChart: Chart = null
let scatterChart: Chart = null
type datasetType = {datasets: {label: string, data: {x: number, y: number}[], backgroundColor: string}[]}
// need separate values since updates replace data field
function getExampleDataSet() {
    return {
        datasets: [{
            label: 'Scatter Dataset',
            data: [{
                x: -10,
                y: 0
            }, {
                x: 0,
                y: 10
            }, {
                x: 10,
                y: 5
            }, {
                x: 0.5,
                y: 5.5
            }],
            backgroundColor: 'rgb(255, 99, 132)'
        }]
    }
}

export function createCharts(kymographCtx: CanvasRenderingContext2D, scatterCtx: CanvasRenderingContext2D) {
    kymographChart = new Chart(kymographCtx, {
        type: 'scatter',
        data: getExampleDataSet(),
        options: {
            animation: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                }
            },
            responsive: false,
            showLine: true,
            elements: {
                point: {
                    radius: 1
                }
            }
        }
    });
    kymographChart.data.datasets[0].label = "Mouse Speed"
    scatterChart = new Chart(scatterCtx, {
        type: 'scatter',
        data: getExampleDataSet(),
        options: {
            animation: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                }
            },
            responsive: false,
            showLine: true,
            elements: {
                point: {
                    radius: 1
                }
            }
        }
    });
    scatterChart.data.datasets[0].label = "Mouse Delta"
}

//https://www.chartjs.org/docs/latest/developers/updates.html - but this actually isn't very good
function addData(chart: Chart, data: ScatterDataPoint[]) {
    chart.data.datasets[0].data = data
    chart.update()
}

export function drawMouseData(kymographCanvas: HTMLCanvasElement,
                              scatterCanvas: HTMLCanvasElement,
                              gameData: GameData, tickData: TickRow, eventData: Row) {
    const parser = gameData.parsers.get(curEvent)
    if (parser != null && parser.havePerTickAimTable && eventData != null) {
        kymographCanvas.style.display = "inline-block"
        scatterCanvas.style.display = "inline-block"
        const aimDataForEvent = gameData.eventToPerTickAimTablesIndices.get(curEvent).get(eventData.id)
        const aimData = gameData.tables.get(parser.perTickAimTable)
        const speedData: ScatterDataPoint[] = []
        const deltaData: ScatterDataPoint[] = []
        for (let i = 0; i < aimDataForEvent.length; i++) {
            const curAimData = aimData[aimDataForEvent[i]]
            if (curAimData.getStartTick() <= tickData.id) {
                speedData.push({
                    x: parseInt(curAimData.otherColumnValues[4]),
                    y: parseFloat(curAimData.otherColumnValues[3])
                })
                deltaData.push({
                    x: parseFloat(curAimData.otherColumnValues[1]),
                    y: parseFloat(curAimData.otherColumnValues[2])
                })
            }
            else {
                break
            }
        }
        addData(kymographChart, speedData)
        addData(scatterChart, deltaData)
    }
    else {
        kymographCanvas.style.display = "none"
        scatterCanvas.style.display = "none"
    }
}
