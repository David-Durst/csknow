import {gameData, parse} from "../data/data";
import {
    tablesNotIndexedByTick,
    gameTableName, Parser,
    ParserType, playerAtTickTableName, playersTableName, RoundRow,
    roundTableName, tablesNotFilteredByRound,
    tickTableName, TickRow, PlayerAtTickRow, parseBool
} from "../data/tables";
import IntervalTree from "@flatten-js/interval-tree";

export let remoteAddr = "http://52.86.105.42:3123/"

export function setRemoteAddr(newRemoteAddr: string) {
    remoteAddr = newRemoteAddr
}

let addedDownloadedOptions = false;
export async function getTables() {
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
                const numForeignKeysIndex = 2
                const numForeignKeys = parseInt(cols[numForeignKeysIndex])
                const numOtherColsIndex = numForeignKeysIndex + numForeignKeys + 1
                const numOtherCols = parseInt(cols[numOtherColsIndex])
                const ticksPerEventIndex = cols.length - 3
                const keyPlayerColumnsIndex = cols.length - 2
                const allTicksIndex = cols.length - 1
                const allTicks = parseBool(cols[allTicksIndex])
                if (allTicks) {
                    gameData.clusters.set(cols[0], [])
                }
                else {
                    gameData.tables.set(cols[0], [])
                }
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
                    new Parser(cols[0], cols[1],
                        cols.slice(numForeignKeysIndex + 1, numForeignKeysIndex + numForeignKeys + 1),
                        cols.slice(numOtherColsIndex + 1, numOtherColsIndex + numOtherCols + 1),
                        cols[ticksPerEventIndex], parserType,
                        remoteAddr + "query/" + cols[0],
                        cols[keyPlayerColumnsIndex], cols[allTicksIndex]
                    )
                )
                if (!tablesNotIndexedByTick.includes(cols[0])) {
                    gameData.ticksToOtherTablesIndices.set(cols[0], new IntervalTree<number>());
                }
                if (!addedDownloadedOptions) {
                    if (allTicks) {
                        (<HTMLSelectElement> document.getElementById("cluster-type"))
                            .add(new Option(cols[0], cols[0]));
                    }
                    else {
                        (<HTMLSelectElement> document.getElementById("event-type"))
                            .add(new Option(cols[0], cols[0]));
                    }
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

export async function getGames() {
    gameData.parsers.get(gameTableName).filterUrl = ""
    await fetch(remoteAddr + "query/games")
        .then((response: Response) => {
            gameData.parsers.get(gameTableName)
                .setReader(response.body.getReader())
            // read first time to
            return gameData.parsers.get(gameTableName).reader.read();
        })
        .then(parse(gameData.parsers.get(gameTableName), true))
        .catch(e => {
            console.log("error downloading " + gameTableName)
            console.log(e)
        })
}

export async function getRounds(gameId: number) {
    gameData.parsers.get(roundTableName).filterUrl = gameId.toString()
    gameData.roundsTable = [];
    await fetch(remoteAddr + "query/rounds/" + gameId.toString())
        .then((response: Response) => {
            gameData.parsers.get(roundTableName)
                .setReader(response.body.getReader())
            // read first time to
            return gameData.parsers.get(roundTableName).reader.read();
        })
        .then(parse(gameData.parsers.get(roundTableName), true))
        .catch(e => {
            console.log("error downloading " + roundTableName)
            console.log(e)
        })
}

export async function getPlayers(gameId: number) {
    gameData.parsers.get(playersTableName).filterUrl = gameId.toString()
    await fetch(remoteAddr + "query/players/" + gameId.toString())
        .then((response: Response) => {
            gameData.parsers.get(playersTableName)
                .setReader(response.body.getReader())
            // read first time to
            return gameData.parsers.get(playersTableName).reader.read();
        })
        .then(parse(gameData.parsers.get(playersTableName), true))
        .catch(e => {
            console.log("error downloading " + playersTableName)
            console.log(e)
        })
}

export function getRoundFilteredTables(promises: Promise<any>[], curRound: RoundRow) {
    for (const downloadedDataName of gameData.tableNames) {
        gameData.parsers.get(downloadedDataName).filterUrl = curRound.gameId.toString()
        if (!tablesNotIndexedByTick.includes(downloadedDataName)) {
            gameData.ticksToOtherTablesIndices.set(downloadedDataName,
                new IntervalTree<number>());
        }
        if (tablesNotFilteredByRound.includes(downloadedDataName)) {
            continue;
        }
        promises.push(
            fetch(remoteAddr + "query/" + downloadedDataName + "/" +
                curRound.id)
                .then((response: Response) => {
                    gameData.parsers.get(downloadedDataName)
                        .setReader(response.body.getReader(),)
                    return gameData.parsers.get(downloadedDataName).reader.read();
                })
                .then(parse(gameData.parsers.get(downloadedDataName), true))
                .catch(e => {
                    console.log("error downloading " + downloadedDataName)
                    console.log(e)
                })
        );
    }
}
