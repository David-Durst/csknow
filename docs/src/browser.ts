import {
    GameData,
    parseKills,
    parsePosition,
    parseSpotted,
    parseWeaponFire,
    parseHurt,
    parseGrenades,
    reader,
    setReader,
    gameData,
    createGameData
} from "./data";

const { S3Client, ListObjectsCommand } = require("@aws-sdk/client-s3");
const {CognitoIdentityClient} = require("@aws-sdk/client-cognito-identity");
const {fromCognitoIdentityPool} = require("@aws-sdk/credential-provider-cognito-identity");
const path = require("path");
import {GetObjectCommand} from "@aws-sdk/client-s3";

const background = new Image();
background.src = "de_dust2_radar_spectate.png";
let canvas: HTMLCanvasElement = null;
let ctx: CanvasRenderingContext2D = null;
let matchSelector: HTMLInputElement = null;
let matchLabel: HTMLLabelElement = null;
const canvasWidth = 700
const canvasHeight = 700
const imageWidth = 1024
const imageHeight = 1024
const black = "rgba(0,0,0,1.0)";
const gray = "rgba(159,159,159,1.0)";
const lightGray = "rgba(200,200,200,0.7)";
const darkBlue = "rgba(4,190,196,1.0)";
const lightBlue = "rgba(194,255,243,1.0)";
const darkRed = "rgba(209,0,0,1.0)";
const lightRed = "rgba(255,143,143,1.0)";

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
    canvas = <HTMLCanvasElement> document.querySelector("#myCanvas");
    ctx = canvas.getContext('2d');
    matchSelector = document.querySelector<HTMLInputElement>("#match-selector")
    matchSelector.value = "0"
    matchSelector.min = "0"
    matchSelector.max = numMatches.toString()
    matchLabel = document.querySelector<HTMLLabelElement>("#cur-match")
    matchLabel.innerHTML = matches[0].demoFile;
    await changedMatch();
    ctx.drawImage(background,0,0,imageWidth,imageHeight,0,0,
        canvasWidth,canvasHeight);
}

function changingMatch() {
    ctx.drawImage(background,0,0,imageWidth,imageHeight,0,0,
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

async function changedMatch() {
    createGameData();
    matchLabel.innerHTML = matches[parseInt(matchSelector.value)].demoFile;
    let promises: Promise<any>[] = []
    try {
        promises.push(
            s3.send(new GetObjectCommand(getObjectParams(matchLabel.innerHTML, "position")))
                .then((response: any) => {
                    setReader(response.Body.getReader());
                    reader.read().then(parsePosition);
                }));
    } catch (err) {
        console.log("Error", err);
    }

    try {
        promises.push(
            s3.send(new GetObjectCommand(getObjectParams(matchLabel.innerHTML, "spotted")))
                .then((response: any) => {
                    setReader(response.Body.getReader());
                    reader.read().then(parseSpotted);
            }));
    } catch (err) {
        console.log("Error", err);
    }

    try {
        promises.push(
            s3.send(new GetObjectCommand(getObjectParams(matchLabel.innerHTML, "weapon_fire")))
                .then((response: any) => {
                    setReader(response.Body.getReader());
                    reader.read().then(parseWeaponFire);
            }));
    } catch (err) {
        console.log("Error", err);
    }

    try {
        promises.push(
            s3.send(new GetObjectCommand(getObjectParams(matchLabel.innerHTML, "hurt")))
                .then((response: any) => {
                    setReader(response.Body.getReader());
                    reader.read().then(parseHurt);
            }));
    } catch (err) {
        console.log("Error", err);
    }

    try {
        promises.push(
            s3.send(new GetObjectCommand(getObjectParams(matchLabel.innerHTML, "grenades")))
                .then((response: any) => {
                    setReader(response.Body.getReader());
                    reader.read().then(parseGrenades);
            }));
    } catch (err) {
        console.log("Error", err);
    }

    try {
        promises.push( 
            s3.send(new GetObjectCommand(getObjectParams(matchLabel.innerHTML, "kills")))
                .then((response: any) => {
                    setReader(response.Body.getReader());
                    reader.read().then(parseKills);
            }));
    } catch (err) {
        console.log("Error", err);
    }
    await Promise.all(promises)
}


export { init, matches, changingMatch, changedMatch, gameData };