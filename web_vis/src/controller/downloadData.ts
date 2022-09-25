import {gameData, parse} from "../data/data";
import {
    gameTableName, Parser,
    ParserType, playerAtTickTableName, playersTableName, RoundRow,
    roundTableName, tablesNotFilteredByRound,
    tickTableName, TickRow, PlayerAtTickRow, parseBool
} from "../data/tables";
import IntervalTree from "@flatten-js/interval-tree";
import {indexEventsForRound} from "../data/ticksToOtherTables";

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
                const ticksPerEventIndex = numOtherColsIndex + numOtherCols + 1
                const keyPlayerColumnsIndex = ticksPerEventIndex + 1
                const nonTemporalIndex = keyPlayerColumnsIndex + 1
                const overlayIndex = nonTemporalIndex + 1
                const overlay = parseBool(cols[overlayIndex])
                const overlayLabelsIndex = overlayIndex + 1
                const overlayLabels = parseBool(cols[overlayLabelsIndex])
                const havePlayerLabelsIndex = overlayLabelsIndex + 1
                const playersToLabelColumnIndex = havePlayerLabelsIndex + 1
                const playerLabelIndicesColumnIndex = playersToLabelColumnIndex + 1
                const playerLabelsIndex = playerLabelIndicesColumnIndex + 1
                const havePerTickAimTable = playerLabelsIndex + 1
                const perTickAimTable = havePerTickAimTable + 1
                const eventIdColumn = perTickAimTable + 1
                if (overlay) {
                    gameData.overlays.set(cols[0], [])
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
                        cols[keyPlayerColumnsIndex], cols[nonTemporalIndex], cols[overlayIndex], cols[overlayLabelsIndex],
                        cols[havePlayerLabelsIndex], cols[playersToLabelColumnIndex], cols[playerLabelIndicesColumnIndex],
                        cols[playerLabelsIndex], cols[havePerTickAimTable], cols[perTickAimTable], cols[eventIdColumn]
                    )
                )
                // the overlay with labels sets the labels for all other overlays
                if (overlayLabels) {
                    gameData.overlayLabels = cols[0]
                }
                if (!addedDownloadedOptions) {
                    if (overlay) {
                        (<HTMLSelectElement> document.getElementById("overlay-type"))
                            .add(new Option(cols[0], cols[0]));
                    }
                    // all tables that are per tick (aka per round and not tick) are events to consider
                    else if (!tablesNotFilteredByRound.includes(cols[0]) && cols[0] != tickTableName){
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

// these are all temporal tables that are filtered by round (round or longer are built in, not dynamically loaded)
export function getRoundFilteredTables(promises: Promise<any>[], curRound: RoundRow) {
    for (const downloadedDataName of gameData.tableNames) {
        if (tablesNotFilteredByRound.includes(downloadedDataName)) {
            continue;
        }
        if (gameData.parsers.get(downloadedDataName).nonTemporal) {
            continue;
        }
        // if temporal, filtered by round, and not jsut the ticks table, setup interval tree for it
        if (downloadedDataName != tickTableName) {
            gameData.ticksToOtherTablesIndices.set(downloadedDataName, new IntervalTree<number>());
        }
        gameData.parsers.get(downloadedDataName).filterUrl = curRound.gameId.toString()
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

export function getNonTemporalTables(promises: Promise<any>[]) {
    for (const downloadedDataName of gameData.tableNames) {
        if (!gameData.parsers.get(downloadedDataName).nonTemporal) {
            continue;
        }
        gameData.parsers.get(downloadedDataName).filterUrl = ""
        promises.push(
            fetch(remoteAddr + "query/" + downloadedDataName)
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
