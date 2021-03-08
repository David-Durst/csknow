import {GameData, Parseable} from "./tables";

const utf8Decoder = new TextDecoder("utf-8");


export function parse(container: Parseable, firstCall: Boolean = true) {
    if (firstCall) {
        container.tempLineContainer = ""
    }
    return async (tuple: ReadableStreamDefaultReadResult<any>) => {
        container.tempLineContainer += tuple.value ?
            utf8Decoder.decode(tuple.value, {stream: true}) : "";
        if (!tuple.done) {
            await container.reader.read().then(parse(container, false));
        }
        else {
            const lines = container.tempLineContainer.split("\n");
            for (let lineNumber = 1; lineNumber < lines.length; lineNumber++) {
                if (lines[lineNumber].trim() === "") {
                    continue;
                }
                container.parseOneLine(lines[lineNumber].split(","))
            }
        }
    }
}

export let gameData: GameData = null;
export function createGameData() {
    gameData = new GameData();
}
export let initialized: boolean = false;
export function setInitialized() {
    initialized = true;
}
