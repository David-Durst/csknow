import HLTV from 'hltv';
const date = require('date-and-time');

const now = new Date();
const nowString = date.format(now, 'YYYY-MM-DD');
const yesterday = new Date();
yesterday.setDate(yesterday.getDate() - 5);
const yesterdayString = date.format(yesterday, 'YYYY-MM-DD');

console.log(yesterdayString)
console.log(nowString)
//HLTV.getPastEvents({ startDate: yesterdayString, endDate: nowString }).then(res => {
//console.log(res)
//})
HLTV.getEvents().then(res => {
    console.log(res)
})
HLTV.getMatches().then(res => {
    console.log(res)
})
