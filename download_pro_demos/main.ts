import HLTV, {GameMap} from 'hltv';
import {FullMatch} from "hltv/lib/endpoints/getMatch";
import * as download from 'download'
import * as fs from 'fs'
import { spawn } from "child_process"

const date = require('date-and-time');

const downloadsFolder = "temp_downloads/"
const now = new Date();
const nowString = date.format(now, 'YYYY-MM-DD');
const yesterday = new Date();
yesterday.setDate(yesterday.getDate() - 1);
const yesterdayString = date.format(yesterday, 'YYYY-MM-DD');

console.log("Downloading demos for " + yesterdayString)
const matchMapStatsIds: number[] = []
const matches: Map<number, FullMatch> = new Map<number, FullMatch>()
const mapStatsIdsToMatchchIds: Map<number, number> = new Map<number, number>()

async function execAsync(command: string, args: string[]) {
    const childProcess = spawn(command, args)
    childProcess.stdout.pipe(process.stdout)
    childProcess.stderr.pipe(process.stderr)
    await new Promise((resolve, reject) => {
        childProcess.addListener("error", reject);
        childProcess.addListener("exit", resolve);
    })
}

async function loadDemos() {
    for (const matchMapStatsId of matchMapStatsIds) {
        if (mapStatsIdsToMatchchIds.has(matchMapStatsId)) {
            console.log("skipping match map " + matchMapStatsId)
        }
        else {
            console.log("getting match map stats id " + matchMapStatsId.toString())
            const matchMapStatsData = await HLTV.getMatchMapStats( { id: matchMapStatsId })
            if (!matches.has(matchMapStatsData.matchId)) {
                await new Promise(r => setTimeout(r, 100));
                const matchData = await HLTV.getMatch({ id: matchMapStatsData.matchId })
                matches.set(matchMapStatsData.matchId, matchData)
                for (const map of matchData.maps) {
                    mapStatsIdsToMatchchIds.set(map.statsId, matchData.id)
                }
                let matchingDemos = 0;
                for (const demo of matchData.demos) {
                    if (demo.name == "GOTV Demo") {
                        matchingDemos++;
                        const rarFile = downloadsFolder + matchData.id.toString() + ".rar"
                        await execAsync("wget", ["-O", rarFile, "https://www.hltv.org/" + demo.link])
                        await execAsync("python3", ["upload_logs.py", matchData.id.toString()])
                    }
                }
                if (matchingDemos != 1) {
                    console.log("matchingDemos " + matchingDemos.toString() +
                        " not 1 for match " + matchData.id)
                }
                console.log("added match " + matchMapStatsData.matchId.toString())
            }
            await new Promise(r => setTimeout(r, 1000));
        }
    }
}

HLTV.getMatchesStats({ startDate: yesterdayString, endDate: yesterdayString,
    delayBetweenPageRequests: 100 , maps: [GameMap.Dust2]}).then((res) => {
    res.map( (matchPreview) =>
        matchMapStatsIds.push(matchPreview.mapStatsId)
    )
    loadDemos()
})


