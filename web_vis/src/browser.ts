import {createGameData, gameData, parse, setInitialized} from "./data/data";
import {
    canvasHeight,
    canvasWidth,
    mainCtx,
    minimap,
    minimapHeight,
    minimapWidth,
    setDemoName,
    setDemoURL,
    setupCanvas,
    setupCanvasHandlers,
    setupMatchDrawing,
    toggleCanvasSize,
    initFilterVars, setupSmallOrLargeMode, smallMode
} from "./drawing/drawing"
import {
    setupFilterHandlers,
    setupMatchFilters,
    filteredData
} from "./controller/filter"
import {registerPlayHandlers} from "./controller/controls"
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
import {indexEventsForRound} from "./data/ticksToOtherTables";
import {
    getGames, getPlayers, getRounds,
    getTables,
    remoteAddr,
    setRemoteAddr,
    getRoundFilteredTables, getNonTemporalTables, getBlob, defaultRemoteAddr
} from "./controller/downloadData";
import {
    setupSelectors,
    matchSelector,
    roundSelector,
    roundLabel, matchLabel, setRoundsSelectorMax
} from "./controller/selectors";

const { S3Client, ListObjectsCommand } = require("@aws-sdk/client-s3");
const {CognitoIdentityClient} = require("@aws-sdk/client-cognito-identity");
const {fromCognitoIdentityPool} = require("@aws-sdk/credential-provider-cognito-identity");
const path = require("path");

let roundLabelStr: string = ""
let downloadSelect: HTMLSelectElement = null;
const black = "rgba(0,0,0,1.0)";
const gray = "rgba(159,159,159,1.0)";
const lightGray = "rgba(200,200,200,0.7)";
const darkBlue = "rgba(4,190,196,1.0)";
const lightBlue = "rgba(194,255,243,1.0)";
const darkRed = "rgba(209,0,0,1.0)";
const lightRed = "rgba(255,143,143,1.0)";

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
    setupSmallOrLargeMode();
    // While loop that runs until response.truncated is false
    downloadSelect = document.querySelector<HTMLSelectElement>("#download-type")
    if (smallMode) {
        setRemoteAddr(path.parse(document.URL).dir + "/bot_example_data/")
    }
    else {
        if (window.location.protocol == "file:") {
            setRemoteAddr("http://localhost:3123/")
        }
        else {
            setRemoteAddr(defaultRemoteAddr)
        }
    }
    setupCanvas()
    createGameData();
    await getTables();
    await getGames();
    await getRounds(0);
    setupSelectors(gameData);
    matchSelector.addEventListener("input", changingMatchOrRound)
    matchSelector.addEventListener("mouseup", changedMatch)
    roundSelector.addEventListener("input", changingMatchOrRound)
    roundSelector.addEventListener("mouseup", changedMatchOrRound)
    await changedMatchOrRound();
    initFilterVars();
    setInitialized();
    registerPlayHandlers();
    document.querySelector<HTMLSelectElement>("#download-type").addEventListener("change", setMatchAndRoundLabels)
    setupCanvasHandlers()
    setupFilterHandlers()
}

function changingMatchOrRound() {
    mainCtx.drawImage(minimap,0,0,minimapWidth,minimapHeight,0,0,
        canvasWidth,canvasHeight);
    mainCtx.fillStyle = lightGray;
    mainCtx.fillRect(0, 0, canvasWidth, canvasHeight)
    setMatchAndRoundLabels();
}

function getObjectParams(key: string, type: string) {
    return {
        Bucket: listBucketParams.Bucket,
        Key: "demos/csvs2/" + key + ".dem_" + type + ".csv"
    };
}


function setMatchAndRoundLabels() {
    roundLabel.innerHTML =
        gameData.roundsTable[parseInt(roundSelector.value)].roundNumber.toString()
    const curGame = gameData.gamesTable[parseInt(matchSelector.value)]
    const demoURL = "https://csknow.s3.amazonaws.com/demos/processed2_small/" + curGame.demoFile
    if (downloadSelect.value == "dem") {
        matchLabel.innerHTML = "<a id=\"match-url\" href=\"" +
             demoURL + "\">" + curGame.demoFile + "</a>"
    }
    else {
        const parser = gameData.parsers.get(downloadSelect.value)
        let url = parser.baseUrl
        if (parser.filterUrl != "") {
            url = url + "/" + parser.filterUrl
        }
        matchLabel.innerHTML = "<a id=\"match-url\" href=\"" + url + "\">" +
            curGame.demoFile + "</a>"
    }
    setDemoURL(demoURL)
    setDemoName(curGame.demoFile)
    /*
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
    setRoundsSelectorMax(gameData.roundsTable.length - 1);
    await changedMatchOrRound();
}


async function changedMatchOrRound() {
    const curGame : GameRow = gameData.gamesTable[parseInt(matchSelector.value)]
    const curRound : RoundRow = gameData.roundsTable[parseInt(roundSelector.value)]
    createGameData();
    setMatchAndRoundLabels();
    let promises: Promise<any>[] = []


    await getPlayers(curGame.id);
    getRoundFilteredTables(promises, curRound);
    getNonTemporalTables(promises);
    getBlob(promises);

    await Promise.all(promises)
    indexEventsForRound(gameData)
    setupMatchFilters()
    setupMatchDrawing()
}


export { init, gameData, filteredData };
