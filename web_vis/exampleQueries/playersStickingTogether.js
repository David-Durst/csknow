let windowTicks = 32*2
function inRange(position, p, otherP) {
    return Math.abs(position.players[p].xPosition - position.players[otherP].xPosition) < 400.0 &&
        Math.abs(position.players[p].yPosition - position.players[otherP].yPosition) < 400.0;
}

let insertedTicks = new Set()

for (let curTick = 0; curTick < localData.position.length; curTick++) {
    let groups = []
    let sourceForGroups = []
    let curPosition = localData.position[curTick]
    for (let p = 0; p < curPosition.players.length; p++) {
        let within4x4 = []
        for (let otherP = p+1; otherP <
        curPosition.players.length; otherP++) {
            if (inRange(curPosition, p, otherP)) {
                within4x4.push(otherP)
            }
        }
        if (within4x4.length >= 2) {
            groups.push(within4x4)
            sourceForGroups.push(p)
        }
    }


    for (let windowTick = curTick; windowTick < curTick + windowTicks &&
    windowTick < localData.position.length; windowTick++) {
        let newPosition = localData.position[windowTick]
        if (curTick == 0) {
            console.log("on tick:" + windowTick.toString())
            console.log(newPosition)
            console.log(groups)
            console.log(sourceForGroups)
        }
        if (groups.length == 0) {
            break;
        }
        let newGroups = []
        let newSources = []
        for (let pSrc = 0; pSrc < sourceForGroups.length; pSrc++) {
            let p = sourceForGroups[pSrc]
            let within4x4 = []
            for (let otherP = 0; otherP < groups[pSrc].length; otherP++) {
                if (inRange(newPosition, p, groups[pSrc][otherP])) {
                    within4x4.push(groups[pSrc][otherP])
                }
            }
            if (within4x4.length >= 4) {
                newGroups.push(within4x4)
                newSources.push(pSrc)
            }
        }
        groups = newGroups
        sourceForGroups = newSources
    }
    if (groups.length >= 1) {
        for (let t = curTick; t < curTick + windowTicks &&
        t < localData.position.length; t++) {
            let tickToAdd = localData.position[t]
            if (!insertedTicks.has(tickToAdd.demoTickNumber)) {
                insertedTicks.add(tickToAdd.demoTickNumber)
                matchingPositions.push(tickToAdd)
            }
        }
    }
}