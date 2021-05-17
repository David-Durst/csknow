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
    GameRow,
    gameTableName,
    Parser,
    ParserType,
    PlayerAtTickRow,
    playerAtTickTableName,
    playersTableName,
    RoundRow,
    roundTableName, tablesNotFilteredByRound,
    TickRow,
    tickTableName
} from "./data/tables";
import {indexEventsForGame} from "./data/ticksToOtherTables";
import {
    getGames, getPlayers, getRounds,
    getTables,
    remoteAddr,
    setRemoteAddr
} from "./controller/downloadData";

const { S3Client, ListObjectsCommand } = require("@aws-sdk/client-s3");
const {CognitoIdentityClient} = require("@aws-sdk/client-cognito-identity");
const {fromCognitoIdentityPool} = require("@aws-sdk/credential-provider-cognito-identity");
const path = require("path");

let matchSelector: HTMLInputElement = null;
let matchLabel: HTMLLabelElement = null;
let roundSelector: HTMLInputElement = null;
let roundLabel: HTMLLabelElement = null;
let roundLabelStr: string = ""
let downloadSelect: HTMLSelectElement = null;
let remoteAddrSelect: HTMLSelectElement = null;
const black = "rgba(0,0,0,1.0)";
const gray = "rgba(159,159,159,1.0)";
const lightGray = "rgba(200,200,200,0.7)";
const darkBlue = "rgba(4,190,196,1.0)";
const lightBlue = "rgba(194,255,243,1.0)";
const darkRed = "rgba(209,0,0,1.0)";
const lightRed = "rgba(255,143,143,1.0)";

function assignRemoteAddr() {
    setRemoteAddr(remoteAddrSelect.value)
}

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
    matchLabel = document.querySelector<HTMLLabelElement>("#cur-match")
    roundLabel = document.querySelector<HTMLLabelElement>("#cur-round")
    downloadSelect = document.querySelector<HTMLSelectElement>("#download-type")
    remoteAddrSelect = document.querySelector<HTMLSelectElement>("#remote-addr")
    setRemoteAddr(remoteAddrSelect.value)
    setupCanvas()
    setupInitFilters()
    createGameData();
    await getTables();
    await getGames();
    await getRounds(0);
    setupMatchRoundSelectors();
    await changedMatchOrRound();
    setInitialized();
    registerPlayHandlers();
    document.querySelector<HTMLSelectElement>("#download-type").addEventListener("change", setMatchLabel)
    document.querySelector<HTMLSelectElement>("#remote-addr").addEventListener("change", assignRemoteAddr)
    matchSelector.addEventListener("input", changingMatchOrRound)
    matchSelector.addEventListener("mouseup", changedMatch)
    roundSelector.addEventListener("input", changingMatchOrRound)
    roundSelector.addEventListener("mouseup", changedMatchOrRound)
    setupCanvasHandlers()
    setupFilterHandlers()
}

function setupMatchRoundSelectors() {
    matchSelector = document.querySelector<HTMLInputElement>("#match-selector")
    matchSelector.value = "0"
    matchSelector.min = "0"
    matchSelector.max = gameData.gamesTable.length.toString()
    roundSelector = document.querySelector<HTMLInputElement>("#round-selector")
    roundSelector.value = "0"
    roundSelector.min = "0"
    roundSelector.max = gameData.gamesTable.length.toString()
}

function changingMatchOrRound() {
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
    console.log("can't set match label yet")
    /*
    setDemoURL("https://csknow.s3.amazonaws.com/demos/processed/" +
        matchLabelStr + ".dem")
    setDemoName(matchLabelStr + ".dem")
    if (downloadSelect.value == "dem") {
        matchLabel.innerHTML = "<a id=\"match-url\" href=\"https://csknow.s3.amazonaws.com/demos/processed/" +
            matchLabelStr + ".dem\">" + matchLabelStr + "</a>"
    }
    else {
        console.log("can't set match label yet")
    }
     */
    /*
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
     */
}

async function changedMatch() {
    await getRounds(gameData.gamesTable[parseInt(matchSelector.value)].id);
    await changedMatchOrRound();
}

async function changedMatchOrRound() {
    const curGame : GameRow = gameData.gamesTable[parseInt(matchSelector.value)]
    const curRound : RoundRow = gameData.roundsTable[parseInt(roundSelector.value)]
    createGameData();
    setMatchLabel();
    let promises: Promise<any>[] = []


    await getPlayers(curGame.id);
    for (const downloadedDataName of gameData.tableNames) {
        if (downloadedDataName in tablesNotFilteredByRound) {
            continue;
        }
        promises.push(
            fetch(remoteAddr + "query/" + downloadedDataName + "/" +
                curRound.id)
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
}


export { init, gameData };
