import {GameData, Parser, Row, TickRow} from "../data/tables";
import {curEvent, displayMouseData} from "./events";
import {Chart, Plugin, PointStyle, registerables, ScatterDataPoint, ScriptableContext} from 'chart.js';
import {AnyObject, EmptyObject} from "chart.js/types/basic";
Chart.register(...registerables);
import annotationPlugin, {AnnotationOptions, PartialEventContext} from 'chartjs-plugin-annotation';
Chart.register(annotationPlugin)

let kymographChart: Chart = null
let scatterChart: Chart = null
type datasetType = {datasets: {label: string, data: {x: number, y: number}[], backgroundColor: string}[]}
// need separate values since updates replace data field
function getExampleDataSet()  {
    return {
        datasets: [
            {
                label: 'Past',
                pointRadius: 1,
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
                backgroundColor: 'rgb(0,213,250)'
            },
            {
                label: 'Present',
                pointRadius: 5,
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
                backgroundColor: 'rgb(255,0,0)'
            },
            {
                label: 'Future',
                pointRadius: 1,
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
                backgroundColor: 'rgb(114,114,114)'
            },
        ]
    }
}

function getLineXPoint(context: PartialEventContext, options: AnnotationOptions): number {
    let dataPoint = context.chart.data.datasets[1].data[0] as ScatterDataPoint
    // take last past point if no current one
    if (dataPoint == null) {
        const length = context.chart.data.datasets[0].data.length
        if (length > 0) {
            dataPoint = context.chart.data.datasets[0].data[length - 1] as ScatterDataPoint
        }
    }
    // take first future point if no current one
    if (dataPoint == null) {
        dataPoint = context.chart.data.datasets[2].data[0] as ScatterDataPoint
    }
    return dataPoint.x
}

const hurtDataIndices: Set<number>  = new Set<number>()
function getPointStyle(context: ScriptableContext<any>, options: AnyObject): PointStyle {
    if (hurtDataIndices.has(context.dataIndex)) {
        return "crossRot"
    }
    else {
        return "circle"
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
                    position: 'bottom',
                    title: {
                        display: true,
                        text: "Time Since Event Start (s)"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "View Angle Speed (deg)"
                    }
                }
            },
            responsive: false,
            showLine: true,
            elements: {
                point: {
                    radius: 1,
                    pointStyle: getPointStyle
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: "View Angle Speed During Event"
                },
                annotation: {
                    annotations: {
                        curTimeLine: {
                            type: 'line',
                            drawTime: 'beforeDatasetsDraw',
                            xMin: getLineXPoint,
                            xMax: getLineXPoint,
                            borderColor: 'rgb(255, 99, 132)',
                            borderWidth: 2,
                        }
                    }
                }
            }
        }
    });
    scatterChart = new Chart(scatterCtx, {
        type: 'scatter',
        data: getExampleDataSet(),
        options: {
            animation: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: "Yaw Delta (deg)"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "Inverted Pitch Delta (deg)"
                    }
                }
            },
            responsive: false,
            showLine: true,
            plugins: {
                title: {
                    display: true,
                    text: "View Angle Delta Relative To Aiming At Enemy Head"
                }
            }
        }
    });
}

//https://www.chartjs.org/docs/latest/developers/updates.html - but this actually isn't very good
function addData(chart: Chart, pastData: ScatterDataPoint[],
                 presentData: ScatterDataPoint[], futureData: ScatterDataPoint[]) {
    chart.data.datasets[0].data = pastData
    chart.data.datasets[1].data = presentData
    chart.data.datasets[2].data = futureData
    chart.config.data.datasets[1].data[1]
    chart.update()
}

export function drawMouseData(kymographCanvas: HTMLCanvasElement,
                              scatterCanvas: HTMLCanvasElement,
                              gameData: GameData, tickData: TickRow, eventData: Row) {
    const parser = gameData.parsers.get(curEvent)
    if (parser != null && parser.havePerTickAimTable && eventData != null && displayMouseData) {
        kymographCanvas.style.display = "inline-block"
        scatterCanvas.style.display = "inline-block"
        const aimDataForEvent = gameData.eventToPerTickAimTablesIndices.get(curEvent).get(eventData.id)
        const aimData = gameData.tables.get(parser.perTickAimTable)
        const pastSpeedData: ScatterDataPoint[] = [], presentSpeedData: ScatterDataPoint[] = [],
            futureSpeedData: ScatterDataPoint[] = []
        const pastDeltaData: ScatterDataPoint[] = [], presentDeltaData: ScatterDataPoint[] = [],
            futureDeltaData: ScatterDataPoint[] = []
        const hurtTickIds = new Set(eventData.otherColumnValues[2].split(";").map(x => parseInt(x)))
        hurtDataIndices.clear()
        for (let i = 0; i < aimDataForEvent.length; i++) {
            const curAimData = aimData[aimDataForEvent[i]]
            let speedData: ScatterDataPoint[] = null
            let deltaData: ScatterDataPoint[] = null
            if (curAimData.getStartTick() < tickData.id) {
                speedData = pastSpeedData
                deltaData = pastDeltaData
            }
            else if (curAimData.getStartTick() == tickData.id) {
                speedData = presentSpeedData
                deltaData = presentDeltaData
            }
            else {
                speedData = futureSpeedData
                deltaData = futureDeltaData
            }
            speedData.push({
                x: parseFloat(curAimData.otherColumnValues[4]),
                y: parseFloat(curAimData.otherColumnValues[3])
            })
            deltaData.push({
                x: parseFloat(curAimData.otherColumnValues[1]),
                y: parseFloat(curAimData.otherColumnValues[2])
            })
            if (hurtTickIds.has(curAimData.foreignKeyValues[0])) {
                hurtDataIndices.add(i)
            }
        }
        addData(kymographChart, pastSpeedData, presentSpeedData, futureSpeedData)
        addData(scatterChart, pastDeltaData, presentDeltaData, futureDeltaData)
    }
    else {
        kymographCanvas.style.display = "none"
        scatterCanvas.style.display = "none"
    }
}
