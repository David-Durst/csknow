import HLTV, {GameMap} from 'hltv';
import {FullMatch} from "hltv/lib/endpoints/getMatch";
import * as download from 'download'
import * as fs from 'fs'

const date = require('date-and-time');

const now = new Date();
const nowString = date.format(now, 'YYYY-MM-DD');
const yesterday = new Date();
yesterday.setDate(yesterday.getDate() - 1);
const yesterdayString = date.format(yesterday, 'YYYY-MM-DD');

console.log("Downloading demos for " + yesterdayString)
//HLTV.getPastEvents({ startDate: yesterdayString, endDate: nowString }).then(res => {
//console.log(res)
//})
/*
HLTV.getEvents().then(res => {
    console.log(res)
})
HLTV.getMatches().then(res => {
    console.log(res)
})

async function saveFile(url: string, dest: string) {
    const file = fs.createWriteStream(dest);
    return new Promise((resolve, reject) =>
            https.get(url, function(response) {
                response.pipe(file);
                file.on('finish', function() {
                    file.close();  // close() is async, call cb after close completes.
                    resolve(1)
                });
            }).on('error', function(err) { // Handle errors
                // Delete the file async. (But we don't check the result)
                fs.unlink(file.path, () => console.log("error in saving " + dest));
                reject(2)
            })
        )
};
 */
const matchMapStatsIds: number[] = []
const matches: Map<number, FullMatch> = new Map<number, FullMatch>()
const mapStatsIdsToMatchchIds: Map<number, number> = new Map<number, number>()

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
                        fs.writeFileSync(matchData.id.toString() + ".rar",
                            await download("https://www.hltv.org/" + demo.link));
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


