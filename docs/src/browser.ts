import {createGameData, gameData, parse, setInitialized} from "./data/data";
import {
    canvasHeight,
    canvasWidth,
    ctx,
    minimap,
    minimapHeight,
    minimapWidth,
    setDemoName,
    setDemoURL,
    setupCanvas,
    setupCanvasHandlers,
    setupMatchDrawing
} from "./drawing/drawing"
import {
    setupFilterHandlers,
    setupInitFilters,
    setupMatchFilters
} from "./controller/filter"
import {registerPlayHandlers} from "./controller/controls"
import {GetObjectCommand, GetObjectCommandOutput} from "@aws-sdk/client-s3";
import {
    gameTableName,
    Parser,
    ParserType,
    PlayerAtTickRow,
    playerAtTickTableName,
    playersTableName,
    RoundRow,
    roundTableName,
    TickRow,
    tickTableName
} from "./data/tables";
import {indexEventsForGame} from "./data/ticksToOtherTables";

const { S3Client, ListObjectsCommand } = require("@aws-sdk/client-s3");
const {CognitoIdentityClient} = require("@aws-sdk/client-cognito-identity");
const {fromCognitoIdentityPool} = require("@aws-sdk/credential-provider-cognito-identity");
const path = require("path");

let matchSelector: HTMLInputElement = null;
let matchLabel: HTMLLabelElement = null;
let matchLabelStr: string = ""
let downloadSelect: HTMLSelectElement = null;
let remoteAddrSelect: HTMLSelectElement = null;
const black = "rgba(0,0,0,1.0)";
const gray = "rgba(159,159,159,1.0)";
const lightGray = "rgba(200,200,200,0.7)";
const darkBlue = "rgba(4,190,196,1.0)";
const lightBlue = "rgba(194,255,243,1.0)";
const darkRed = "rgba(209,0,0,1.0)";
const lightRed = "rgba(255,143,143,1.0)";

let remoteAddr = "http://52.86.105.42:3123/"
function setRemoteAddr() {
    remoteAddr = remoteAddrSelect.value
}

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
    remoteAddrSelect = document.querySelector<HTMLSelectElement>("#remote-addr")
    remoteAddr = remoteAddrSelect.value
    setupCanvas()
    setupInitFilters()
    await changedMatch();
    setInitialized();
    registerPlayHandlers();
    document.querySelector<HTMLSelectElement>("#download-type").addEventListener("change", setMatchLabel)
    document.querySelector<HTMLSelectElement>("#remote-addr").addEventListener("change", setRemoteAddr)
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
    setDemoURL("https://csknow.s3.amazonaws.com/demos/processed/" +
        matchLabelStr + ".dem")
    setDemoName(matchLabelStr + ".dem")
    if (downloadSelect.value == "dem") {
        matchLabel.innerHTML = "<a id=\"match-url\" href=\"https://csknow.s3.amazonaws.com/demos/processed/" +
            matchLabelStr + ".dem\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "position") {
        matchLabel.innerHTML = "<a id=\"match-url\" href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_position.csv\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "spotted") {
        matchLabel.innerHTML = "<a id=\"match-url\" href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_spotted.csv\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "weapon_fire") {
        matchLabel.innerHTML = "<a id=\"match-url\" href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_weapon_fire.csv\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "hurt") {
        matchLabel.innerHTML = "<a id=\"match-url\" href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_hurt.csv\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "grenades") {
        matchLabel.innerHTML = "<a id=\"match-url\" href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_grenades.csv\">" + matchLabelStr + "</a>"
    }
    else if (downloadSelect.value == "kills") {
        matchLabel.innerHTML = "<a id=\"match-url\" href=\"https://csknow.s3.amazonaws.com/demos/csvs2/" +
            matchLabelStr + ".dem_kills.csv\">" + matchLabelStr + "</a>"
    }
    else {
        matchLabel.innerHTML = "<a id=\"match-url\" href=\"" + remoteAddr + "query/" + downloadSelect.value + "/" +
            matchLabelStr + ".dem.csv\">" + matchLabelStr + "</a>"
    }
}

let addedDownloadedOptions = false;
async function changedMatch() {
    createGameData();
    matchLabelStr = matches[parseInt(matchSelector.value)].demoFile;
    setMatchLabel();
    let promises: Promise<any>[] = []

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
                gameData.tableNames.push(cols[0])
                gameData.tables.set(cols[0], [])
                const numKeysIndex = 1
                const numKeys = parseInt(cols[numKeysIndex])
                const numOtherColsIndex = numKeysIndex + numKeys + 1
                const numOtherCols = parseInt(cols[numOtherColsIndex])
                let parserType: ParserType;
                if (cols[0] == tickTableName) {
                    parserType = ParserType.tick;
                }
                else if (cols[0] == gameTableName) {
                    parserType = ParserType.game;
                }
                else if (cols[0] == roundTableName) {
                    parserType = ParserType.round;
                }
                else if (cols[0] == playerAtTickTableName) {
                    parserType = ParserType.playerAtTick;
                }
                else if (cols[0] == playersTableName) {
                    parserType = ParserType.player;
                }
                else {
                    parserType = ParserType.other
                }
                gameData.parsers.set(cols[0],
                    new Parser(cols[0],
                        cols.slice(numKeysIndex + 1, numKeysIndex + numKeys + 1),
                        cols.slice(numOtherColsIndex + 1, numOtherColsIndex + numOtherCols + 1),
                        cols[cols.length - 1], parserType
                    )
                )
                gameData.ticksToOtherTablesIndices.set(cols[0], new Map<number, number[]>());
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

    for (const downloadedDataName of gameData.tableNames) {
        promises.push(
            fetch(remoteAddr + "query/" + downloadedDataName + "/" +
                matches[parseInt(matchSelector.value)].demoFileWithExt + ".csv")
            .then((response: Response) => {
                gameData.parsers.get(downloadedDataName)
                    .setReader(response.body.getReader(), )
                return gameData.parsers.get(downloadedDataName).reader.read();
            })
            .then(parse(gameData.parsers.get(downloadedDataName), true))
            .catch(e => {
                console.log("error downloading " + downloadedDataName)
            })
        );
    }
    gameData.roundsTable = <RoundRow[]> gameData.tables.get(roundTableName)
    gameData.tables.delete(roundTableName)
    gameData.ticksTable = <TickRow[]> gameData.tables.get(tickTableName)
    gameData.tables.delete(tickTableName)
    gameData.playerAtTicksTable =
        <PlayerAtTickRow[]> gameData.tables.get(playerAtTickTableName)


    await Promise.all(promises)
    indexEventsForGame(gameData)
    setupMatchFilters()
    setupMatchDrawing()
    addedDownloadedOptions = true;
}


export { init, matches, gameData };
