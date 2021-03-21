import {
    parse,
    gameData,
    createGameData,
    initialized, setInitialized
} from "./data/data";
import {
    canvas,
    ctx,
    canvasWidth,
    canvasHeight,
    minimapWidth,
    minimapHeight,
    minimapScale,
    minimap,
    drawTick,
    setupMatchDrawing,
    setupCanvas, setupCanvasHandlers
} from "./drawing/drawing"
import {
    setupFilterHandlers,
    setupInitFilters, setupMatchFilters
} from "./controller/filter"
import { registerPlayHandlers } from "./controller/controls"

const { S3Client, ListObjectsCommand } = require("@aws-sdk/client-s3");
const {CognitoIdentityClient} = require("@aws-sdk/client-cognito-identity");
const {fromCognitoIdentityPool} = require("@aws-sdk/credential-provider-cognito-identity");
const path = require("path");
import {
    GetObjectCommand,
    GetObjectCommandOutput,
    GetObjectOutput
} from "@aws-sdk/client-s3";
import {DownloadParser, GameData, setReader} from "./data/tables";
import {indexEventsForGame} from "./data/positionsToEvents";

let matchSelector: HTMLInputElement = null;
let matchLabel: HTMLLabelElement = null;
let matchLabelStr: string = ""
let downloadSelect: HTMLSelectElement = null;
const black = "rgba(0,0,0,1.0)";
const gray = "rgba(159,159,159,1.0)";
const lightGray = "rgba(200,200,200,0.7)";
const darkBlue = "rgba(4,190,196,1.0)";
const lightBlue = "rgba(194,255,243,1.0)";
const darkRed = "rgba(209,0,0,1.0)";
const lightRed = "rgba(255,143,143,1.0)";
const remoteAddr = "http://52.86.105.42:3123/"

class Match {
    key: any
    demoFileWithExt: string;
    demoFile: string

    constructor(key: any) {
        this.key = key
        const p = path.parse(key)
        this.demoFileWithExt = p.base
        this.demoFile = p.name
    }

}
let matches: Match[] = [];

// Set the AWS region
const REGION = "us-east-1"; //e.g. "us-east-1"

// Create the parameters for the bucket
const listBucketParams = {
    Bucket: "csknow",
    Prefix: "demos/processed/auto",
    // @ts-ignore
    Marker: undefined
};

const cognitoIdentityClient = new CognitoIdentityClient({
    region: REGION
});

// Create S3 service object
const s3 = new S3Client({
    region: REGION,
    credentials: fromCognitoIdentityPool({
        client: cognitoIdentityClient,
        identityPoolId: "us-east-1:b97cc6dd-33b3-4672-a48b-9fa1876d8c78"
    })
});

async function init() {
    // Declare truncated as a flag that we will base our while loop on
    let truncated = true;
    // Declare a variable that we will assign the key of the last element in the response to
    let pageMarker;
    // While loop that runs until response.truncated is false
    let numMatches = 0;
    while (truncated) {
        try {
            const response = await s3.send(new ListObjectsCommand(listBucketParams));
            response.Contents.forEach((item: any) => {
                matches.push(new Match(item.Key))
            })
            numMatches += response.Contents.length
            truncated = response.IsTruncated;
            // If 'truncated' is true, assign the key of the final element in the response to our variable 'pageMarker'
            if (truncated) {
                pageMarker = response.Contents.slice(-1)[0].Key;
                // Assign value of pageMarker to bucketParams so that the next iteration will start from the new pageMarker.
                listBucketParams.Marker = pageMarker;
            }
            // At end of the list, response.truncated is false and our function exits the while loop.
        } catch (err) {
            console.log("Error", err);
            truncated = false;
        }
    }
    matchSelector = document.querySelector<HTMLInputElement>("#match-selector")
    matchSelector.value = "0"
    matchSelector.min = "0"
    matchSelector.max = numMatches.toString()
    matchLabel = document.querySelector<HTMLLabelElement>("#cur-match")
    downloadSelect = document.querySelector<HTMLSelectElement>("#download-type")
    setupCanvas()
    setupInitFilters()
    await changedMatch();
    setInitialized();
    registerPlayHandlers();
    document.querySelector<HTMLSelectElement>("#download-type").addEventListener("change", setMatchLabel)
    matchSelector.addEventListener("input", changingMatch)
    matchSelector.addEventListener("mouseup", changedMatch)
    setupCanvasHandlers()
    setupFilterHandlers()
}

function changingMatch() {
    ctx.drawImage(minimap,0,0,minimapWidth,minimapHeight,0,0,
        canvasWidth,canvasHeight);
    ctx.fillStyle = lightGray;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight)
}

function getObjectParams(key: string, type: string) {
    return {
        Bucket: listBucketParams.Bucket,
        Key: "demos/csvs2/" + key + ".dem_" + type + ".csv"
    };
}

function setMatchLabel() {
    if (downloadSelect.value == "dem") {
        matchLabel.innerHTML = "<a href=\"https://csknow.s3.amazonaws.com/demos/processed/" +
            matchLabelStr + ".dem\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "position") {
        matchLabel.innerHTML = "<a href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_position.csv\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "spotted") {
        matchLabel.innerHTML = "<a href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_spotted.csv\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "weapon_fire") {
        matchLabel.innerHTML = "<a href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_weapon_fire.csv\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "hurt") {
        matchLabel.innerHTML = "<a href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_hurt.csv\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "grenades") {
        matchLabel.innerHTML = "<a href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_grenades.csv\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "kills") {
        matchLabel.innerHTML = "<a href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_kills.csv\">" + matchLabelStr + "</a>"
    }
    else {
        matchLabel.innerHTML = "<a href=\"" + remoteAddr + "query/" + downloadSelect.value + "/" +
            matchLabelStr + ".dem.csv\">" + matchLabelStr + "</a>"
    }
}

let addedDownloadedOptions = false;
async function changedMatch() {
    createGameData();
    matchLabelStr = matches[parseInt(matchSelector.value)].demoFile;
    setMatchLabel();
    let promises: Promise<any>[] = []
    try {
        promises.push(
            s3.send(new GetObjectCommand(getObjectParams(matchLabelStr, "position")))
                .then((response: any) => {
                    setReader((<ReadableStream> response.Body).getReader(), gameData.positionParser)
                    return gameData.positionParser.reader.read();
                }).then(parse(gameData.positionParser, true)));
    } catch (err) {
        console.log("Error", err);
    }

    try {
        promises.push(
            s3.send(new GetObjectCommand(getObjectParams(matchLabelStr, "spotted")))
                .then((response: any) => {
                    setReader((<ReadableStream> response.Body).getReader(), gameData.spottedParser)
                    return gameData.spottedParser.reader.read();
            }).then(parse(gameData.spottedParser, true)));
    } catch (err) {
        console.log("Error", err);
    }

    try {
        promises.push(
            s3.send(new GetObjectCommand(getObjectParams(matchLabelStr, "weapon_fire")))
                .then((response: GetObjectCommandOutput) => {
                    setReader((<ReadableStream> response.Body).getReader(), gameData.weaponFireParser)
                    return gameData.weaponFireParser.reader.read();
            }).then(parse(gameData.weaponFireParser, true)));
    } catch (err) {
        console.log("Error", err);
    }

    try {
        promises.push(
            s3.send(new GetObjectCommand(getObjectParams(matchLabelStr, "hurt")))
                .then((response: GetObjectCommandOutput) => {
                    setReader((<ReadableStream> response.Body).getReader(), gameData.playerHurtParser)
                    return gameData.playerHurtParser.reader.read();
            }).then(parse(gameData.playerHurtParser, true)));
    } catch (err) {
        console.log("Error", err);
    }

    try {
        promises.push(
            s3.send(new GetObjectCommand(getObjectParams(matchLabelStr, "grenades")))
                .then((response: GetObjectCommandOutput) => {
                    setReader((<ReadableStream> response.Body).getReader(), gameData.grenadeParser)
                    return gameData.grenadeParser.reader.read();
            }).then(parse(gameData.grenadeParser, true)));
    } catch (err) {
        console.log("Error", err);
    }

    try {
        promises.push(
            s3.send(new GetObjectCommand(getObjectParams(matchLabelStr, "kills")))
                .then((response: GetObjectCommandOutput) => {
                    setReader((<ReadableStream> response.Body).getReader(), gameData.killsParser)
                    return gameData.killsParser.reader.read();
            }).then(parse(gameData.killsParser, true)));
    } catch (err) {
        console.log("Error", err);
    }

    // wait for list of responses, then do each element
    await fetch(remoteAddr + "list", {mode: 'no-cors'})
        .then(_ =>
            fetch(remoteAddr + "list")
        )
        .then((response: Response) =>
            response.text()
        )
        .then((remoteTablesText: string) => {
            const lines = remoteTablesText.trim().split("\n");
            for (let lineNumber = 0; lineNumber < lines.length; lineNumber++) {
                const cols = lines[lineNumber].split(",");
                gameData.downloadedDataNames.push(cols[0])
                gameData.downloadedData.set(cols[0], [])
                const numTargetsIndex = 2
                const numTargets = parseInt(cols[numTargetsIndex])
                const numOtherColsIndex = numTargetsIndex + numTargets + 1
                const numOtherCols = parseInt(cols[numOtherColsIndex])
                gameData.downloadedParsers.set(cols[0],
                    new DownloadParser(cols[0], cols[1],
                        cols.slice(numTargetsIndex + 1, numTargetsIndex + numTargets + 1),
                        cols.slice(numOtherColsIndex + 1, numOtherColsIndex + numOtherCols + 1),
                        parseInt(cols[cols.length - 1])
                    )
                )
                gameData.downloadedPositionToEvent.set(cols[0], new Map<number, number[]>());
                if (!addedDownloadedOptions) {
                    (<HTMLSelectElement> document.getElementById("event-type"))
                        .add(new Option(cols[0], cols[0]));
                    (<HTMLSelectElement> document.getElementById("download-type"))
                        .add(new Option(cols[0], cols[0]));
                }
            }
        })
        .catch(e => {
            console.log("can't read listing from remote server")
            console.log(e)
        });

    for (const downloadedDataName of gameData.downloadedDataNames) {
        promises.push(
            fetch(remoteAddr + "query/" + downloadedDataName + "/" +
                matches[parseInt(matchSelector.value)].demoFileWithExt + ".csv")
            .then((response: Response) => {
                setReader(response.body.getReader(), gameData.downloadedParsers.get(downloadedDataName))
                return gameData.downloadedParsers.get(downloadedDataName).reader.read();
            })
            .then(parse(gameData.downloadedParsers.get(downloadedDataName), true))
            .catch(e => {
                console.log("error downloading " + downloadedDataName)
            })
        );
    }

    await Promise.all(promises)
    indexEventsForGame(gameData)
    setupMatchFilters()
    setupMatchDrawing()
    addedDownloadedOptions = true;
}


export { init, matches, gameData };
