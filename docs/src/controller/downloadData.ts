import {gameData} from "../data/data";
import {
    customParsedTabled,
    gameTableName, Parser,
    ParserType, playerAtTickTableName, playersTableName,
    roundTableName,
    tickTableName
} from "../data/tables";

export let remoteAddr = "http://52.86.105.42:3123/"

export function setRemoteAddr(newRemoteAddr: string) {
    remoteAddr = newRemoteAddr
}

let addedDownloadedOptions = false;
async function getTables() {
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
                const numForeignKeysIndex = 1
                const numForeignKeys = parseInt(cols[numForeignKeysIndex])
                const numOtherColsIndex = numForeignKeysIndex + numForeignKeys + 1
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
                        cols.slice(numForeignKeysIndex + 1, numForeignKeysIndex + numForeignKeys + 1),
                        cols.slice(numOtherColsIndex + 1, numOtherColsIndex + numOtherCols + 1),
                        cols[cols.length - 1], parserType
                    )
                )
                if (!(cols[0] in customParsedTabled)) {
                    gameData.ticksToOtherTablesIndices.set(cols[0], new Map<number, number[]>());
                }
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
    addedDownloadedOptions = true;
}