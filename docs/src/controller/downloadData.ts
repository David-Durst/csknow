import {gameData, parse} from "../data/data";
import {
    customParsedTables,
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
                if (!(cols[0] in customParsedTables)) {
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

export async function getGames() {
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
        })
}

export async function getRounds(gameId: number) {
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
        })
}

export async function getPlayers(gameId: number) {
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
        })
}
