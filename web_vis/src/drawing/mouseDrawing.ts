import {GameData, Parser, Row, TickRow} from "../data/tables";
import {curEvent} from "./events";
import {Chart, registerables, ScatterDataPoint} from 'chart.js';
Chart.register(...registerables);

let kymographChart: Chart = null
let scatterChart: Chart = null
type datasetType = {datasets: {label: string, data: {x: number, y: number}[], backgroundColor: string}[]}
const exampleDataSet = {
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
    }],
};
export function createCharts(kymographCtx: CanvasRenderingContext2D, scatterCtx: CanvasRenderingContext2D) {
    kymographChart = new Chart(kymographCtx, {
        type: 'scatter',
        data: exampleDataSet,
        options: {
            animation: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                }
            }
        }
    });
    kymographChart.data.datasets[0].label = "Mouse Speed"
    scatterChart = new Chart(scatterCtx, {
        type: 'scatter',
        data: exampleDataSet,
        options: {
            animation: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom'
                }
            }
        }
    });
    scatterChart.data.datasets[0].label = "Mouse Delta"
}

//https://www.chartjs.org/docs/latest/developers/updates.html
function addData(chart: Chart, data: ScatterDataPoint[]) {
    for (const dataPoint of data) {
        chart.data.datasets[0].data.push(dataPoint)
    }
}

function removeData(chart: Chart) {
    chart.data.datasets.forEach((dataset) => {
        dataset.data.pop();
    });
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
                    x: parseInt(eventData.otherColumnValues[4]),
                    y: parseInt(eventData.otherColumnValues[3])
                })
                deltaData.push({
                    x: parseInt(eventData.otherColumnValues[1]),
                    y: parseInt(eventData.otherColumnValues[2])
                })
            }
            else {
                break
            }
        }
        removeData(kymographChart)
        addData(kymographChart, speedData)
        removeData(scatterChart)
        addData(scatterChart, deltaData)
    }
    else {
        kymographCanvas.style.display = "none"
        scatterCanvas.style.display = "none"
    }
}
