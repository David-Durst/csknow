package internal

import (
	"fmt"
	c "github.com/David-Durst/csknow/demo_parser/internal/constants"
	"github.com/golang/geo/r3"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/events"
	"os"
	"path/filepath"
	"strings"
)

type RowIndex int64

const InvalidId RowIndex = -1
const InvalidInt int = -1

func boolToInt(b bool) int {
	if b {
		return 1
	} else {
		return 0
	}
}

// IDState the next id to use for each fact table
type IDState struct {
	nextGame              RowIndex
	nextPlayer            RowIndex
	nextRound             RowIndex
	nextTick              RowIndex
	nextPlayerAtTick      RowIndex
	nextSpotted           RowIndex
	nextFootstep          RowIndex
	nextWeaponFire        RowIndex
	nextKill              RowIndex
	nextPlayerHurt        RowIndex
	nextGrenade           RowIndex
	nextGrenadeTrajectory RowIndex
	nextPlayerFlashed     RowIndex
	nextPlant             RowIndex
	nextDefusal           RowIndex
	nextExplosion         RowIndex
}

func DefaultIDState() IDState {
	return IDState{0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0}
}

type tableRow interface {
	toString() string
}

type table[T tableRow] struct {
	rows []T
}

func (t *table[T]) tail() *T {
	return &t.rows[len(t.rows)-1]
}

func (t *table[T]) get(i RowIndex) *T {
	return &t.rows[i]
}

func (t *table[T]) append(e T) {
	t.rows = append(t.rows, e)
}

func (t *table[T]) len() int {
	return len(t.rows)
}

func (t *table[T]) saveToFile(fileName string, header string) {
	file, err := os.Create(filepath.Join(c.TmpDir, fileName))
	if err != nil {
		panic(err)
	}
	defer file.Close()
	var sb strings.Builder
	sb.WriteString(header)
	for _, element := range t.rows {
		sb.WriteString(element.toString())
	}
	file.WriteString(sb.String())
}

// GAMES TABLE

const gamesHeader = "id,demo_file,demo_tick_rate,game_tick_rate,map_name,game_type\n"

type gameRow struct {
	id           RowIndex
	demoFile     string
	demoTickRate float64
	gameTickRate float64
	mapName      string
	gameType     c.GameType
}

func (g gameRow) toString() string {
	return fmt.Sprintf("%d,%s,%f,%f,%s,%d\n",
		g.id, g.demoFile, g.demoTickRate, g.gameTickRate, g.mapName, g.gameType)
}

var curGameRow gameRow

// PLAYERS TABLE

const playersHeader = "id,game_id,name,steam_id\n"

type playerRow struct {
	id      RowIndex
	gameId  RowIndex
	name    string
	steamID uint64
}

var defaultPlayer = playerRow{InvalidId, InvalidId, "invalid", 0}

func (p playerRow) toString() string {
	return fmt.Sprintf("%d,%d,%s,%d\n", p.id, p.gameId, p.name, p.steamID)
}

var playersTable table[playerRow]

type playersTrackerT struct {
	gameIdToTableId map[int]RowIndex
}

func (p *playersTrackerT) init() {
	p.gameIdToTableId = make(map[int]RowIndex)
}

func (p *playersTrackerT) addPlayer(pr playerRow, gameUserId int) {
	p.gameIdToTableId[gameUserId] = pr.id
	playersTable.append(pr)
}

func (p *playersTrackerT) alreadyAddedPlayer(gameUserId int) bool {
	_, ok := p.gameIdToTableId[gameUserId]
	return ok
}

func (p *playersTrackerT) getPlayerIdFromGameData(player *common.Player) RowIndex {
	if tableId, ok := p.gameIdToTableId[player.UserID]; ok {
		return tableId
	} else {
		return InvalidId
	}
}

var playersTracker playersTrackerT

// ROUNDS TABLE

const roundsHeader = "id,game_id,start_tick,end_tick,end_official_tick,warmup,overtime," +
	"freeze_time_end,round_number,round_end_reason,winner,t_wins,ct_wins\n"

type roundRow struct {
	finished        bool
	id              RowIndex
	gameId          RowIndex
	startTick       RowIndex
	endTick         RowIndex
	endOfficialTick RowIndex
	warmup          bool
	overtime        bool
	freezeTimeEnd   RowIndex
	roundNumber     int
	roundEndReason  int
	winner          common.Team
	tWins           int
	ctWins          int
}

var defaultRound = roundRow{false, InvalidId, InvalidId, InvalidId, InvalidId,
	InvalidId, true, false, InvalidId, 0, InvalidInt,
	common.TeamUnassigned, InvalidInt, InvalidInt}

func (r roundRow) toString() string {
	return fmt.Sprintf(
		"%d,%d,%d,%d,%d,%d,%d,"+
			"%d,%d,%d,%d,%d,%d\n",
		r.id, r.gameId, r.startTick, r.endTick, r.endOfficialTick, boolToInt(r.warmup), boolToInt(r.overtime),
		r.freezeTimeEnd, r.roundNumber, r.roundEndReason, r.winner, r.tWins, r.ctWins)
}

var unfilteredRoundsTable table[roundRow]
var filteredRoundsTable table[roundRow]

// TICKS TABLE

const ticksHeader = "id,round_id,game_time,demo_tick_number,game_tick_number,bomb_carrier,bomb_x,bomb_y,bomb_z\n"

type tickRow struct {
	id             RowIndex
	roundId        RowIndex
	gameTime       int64
	demoTickNumber int
	gameTickNumber int
	bombCarrier    RowIndex
	bombX          float64
	bombY          float64
	bombZ          float64
}

func (t tickRow) toString() string {
	return fmt.Sprintf("%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f\n",
		t.id, t.roundId, t.gameTime, t.demoTickNumber, t.gameTickNumber,
		t.bombCarrier, t.bombX, t.bombY, t.bombZ)
}

var ticksTable table[tickRow]

// PLAYERATTICKS TABLE

const playerAtTicksHeader = "id,player_id,tick_id,pos_x,pos_y,pos_z,eye_pos_z,vel_x,vel_y,vel_z,view_x,view_y,aim_punch_x,aim_punch_y," +
	"view_punch_x,view_punch_y,team,health,armor,has_helmet,is_alive,ducking_key_pressed,duck_amount,is_walking,is_scoped," +
	"is_airborne,remaining_flash_time,active_weapon,main_weapon,primary_bullets_clip," +
	"primary_bullets_reserve,secondary_weapon,secondary_bullets_clip,secondary_bullets_reserve,num_he," +
	"num_flash,num_smoke,num_incendiary,num_molotov,num_decoy,num_zeus,has_defuser,has_bomb,money,ping\n"

type playerAtTickRow struct {
	id                      RowIndex
	playerId                RowIndex
	tickId                  RowIndex
	posX                    float64
	posY                    float64
	posZ                    float64
	eyePosZ                 float64
	velX                    float64
	velY                    float64
	velZ                    float64
	viewX                   float32
	viewY                   float32
	aimPunchX               float64
	aimPunchY               float64
	viewPunchX              float64
	viewPunchY              float64
	team                    int
	health                  int
	armor                   int
	hasHelmet               bool
	isAlive                 bool
	duckingKeyPressed       bool
	duckAmount              float32
	isWalking               bool
	isScoped                bool
	isAirborne              bool
	flashDuration           float32
	activeWeapon            common.EquipmentType
	primaryWeapon           common.EquipmentType
	primaryBulletsClip      int
	primaryBulletsReserve   int
	secondaryWeapon         common.EquipmentType
	secondaryBulletsClip    int
	secondaryBulletsReserve int
	numHE                   int
	numFlash                int
	numSmoke                int
	numIncendiary           int
	numMolotov              int
	numDecoy                int
	numZeus                 int
	hasDefuser              bool
	hasBomb                 bool
	money                   int
	ping                    int
}

func (p playerAtTickRow) toString() string {
	return fmt.Sprintf(
		"%d,%d,%d,%.2f,%.2f,%.2f,%.2f,"+
			"%.2f,%.2f,%.2f,"+
			"%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"+
			"%d,%d,%d,%d,"+
			"%d,%d,%.2f,%d,%d,%d,"+
			"%f,%d,%d,%d,"+
			"%d,%d,%d,%d,"+
			"%d,%d,%d,%d,%d,%d,%d,"+
			"%d,%d,%d,%d\n",
		p.id, p.playerId, p.tickId, p.posX, p.posY, p.posZ, p.eyePosZ,
		p.velX, p.velY, p.velZ,
		p.viewX, p.viewY, p.aimPunchX, p.aimPunchY, p.viewPunchX, p.viewPunchY,
		p.team, p.health, p.armor, boolToInt(p.hasHelmet),
		boolToInt(p.isAlive), boolToInt(p.duckingKeyPressed), p.duckAmount, boolToInt(p.isWalking), boolToInt(p.isScoped), boolToInt(p.isAirborne),
		p.flashDuration, p.activeWeapon, p.primaryWeapon, p.primaryBulletsClip,
		p.primaryBulletsReserve, p.secondaryWeapon, p.secondaryBulletsClip, p.secondaryBulletsReserve,
		p.numHE, p.numFlash, p.numSmoke, p.numIncendiary, p.numMolotov, p.numDecoy, p.numZeus,
		boolToInt(p.hasDefuser), boolToInt(p.hasBomb), p.money, p.ping)
}

var playerAtTicksTable table[playerAtTickRow]

// SPOTTED TABLE

const spottedHeader = "id,tick_id,spotted_player,spotter_player,is_spotted\n"

type spottedRow struct {
	id            RowIndex
	tickId        RowIndex
	spottedPlayer RowIndex
	spotterPlayer RowIndex
	isSpotted     bool
}

func (s spottedRow) toString() string {
	return fmt.Sprintf("%d,%d,%d,%d,%d\n",
		s.id, s.tickId, s.spottedPlayer, s.spotterPlayer, boolToInt(s.isSpotted))
}

var spottedTable table[spottedRow]

// FOOTSTEP TABLE

const footstepHeader = "id,tick_id,stepping_player\n"

type footstepRow struct {
	id             RowIndex
	tickId         RowIndex
	steppingPlayer RowIndex
}

func (f footstepRow) toString() string {
	return fmt.Sprintf("%d,%d,%d\n", f.id, f.tickId, f.steppingPlayer)
}

var footstepTable table[footstepRow]

// WEAPONFIRE TABLE

const weaponFireHeader = "id,tick_id,shooter,weapon\n"

type weaponFireRow struct {
	id      RowIndex
	tickId  RowIndex
	shooter RowIndex
	weapon  common.EquipmentType
}

func (w weaponFireRow) toString() string {
	return fmt.Sprintf("%d,%d,%d,%d\n", w.id, w.tickId, w.shooter, w.weapon)
}

var weaponFireTable table[weaponFireRow]

// HURT TABLE

const hurtHeader = "id,tick_id,victim,attacker,weapon,armor_damage,armor,health_damage,health,hit_group\n"

type hurtRow struct {
	id           RowIndex
	tickId       RowIndex
	victim       RowIndex
	attacker     RowIndex
	weapon       common.EquipmentType
	armorDamage  int
	armor        int
	healthDamage int
	health       int
	hitGroup     events.HitGroup
}

func (h hurtRow) toString() string {
	return fmt.Sprintf(
		"%d,%d,%d,%d,"+
			"%d,%d,%d,%d,%d,%d\n",
		h.id, h.tickId, h.victim, h.attacker,
		h.weapon, h.armorDamage, h.armor, h.healthDamage, h.health, h.hitGroup)
}

var hurtTable table[hurtRow]

// KILL TABLE

const killHeader = "id,tick_id,killer,victim,weapon,assister,is_headshot,is_wallbang,penetrated_objects\n"

type killRow struct {
	id                RowIndex
	tickId            RowIndex
	killer            RowIndex
	victim            RowIndex
	weapon            common.EquipmentType
	assister          RowIndex
	isHeadshot        bool
	isWallbang        bool
	penetratedObjects int
}

func (h killRow) toString() string {
	return fmt.Sprintf(
		"%d,%d,%d,%d,%d,"+
			"%d,%d,%d,%d\n",
		h.id, h.tickId, h.killer, h.victim, h.weapon,
		h.assister, boolToInt(h.isHeadshot), boolToInt(h.isWallbang), h.penetratedObjects)
}

var killTable table[killRow]

// GRENADE TABLE

const grenadeHeader = "id,thrower,grenade_type,throw_tick,active_tick,expired_tick,destroy_tick\n"

type grenadeRow struct {
	id          RowIndex
	thrower     RowIndex
	grenadeType common.EquipmentType
	throwTick   RowIndex
	activeTick  RowIndex
	expiredTick RowIndex
	destroyTick RowIndex
	trajectory  []r3.Vector
}

func (g grenadeRow) toString() string {
	return fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d,%d\n",
		g.id, g.thrower, g.grenadeType, g.throwTick, g.activeTick, g.expiredTick, g.expiredTick, g.destroyTick)
}

var grenadeTable table[grenadeRow]

type grenadeTrackerT struct {
	uniqueIdToTableId map[int64]RowIndex
}

func (g *grenadeTrackerT) init() {
	g.uniqueIdToTableId = make(map[int64]RowIndex)
}

func (g *grenadeTrackerT) addGrenade(gr grenadeRow, uniqueId int64) {
	g.uniqueIdToTableId[uniqueId] = gr.id
	grenadeTable.append(gr)
}

func (g *grenadeTrackerT) alreadyAddedGrenade(uniqueId int64) bool {
	_, ok := g.uniqueIdToTableId[uniqueId]
	return ok
}

func (g *grenadeTrackerT) getGrenadeIdFromGameData(uniqueId int64) RowIndex {
	if tableId, ok := g.uniqueIdToTableId[uniqueId]; ok {
		return tableId
	} else {
		return InvalidId
	}
}

var grenadeTracker grenadeTrackerT

// GRENADETRAJECTORY TABLE

const grenadeTrajectoryHeader = "id,grenade_id,id_per_grenade,pos_x,pos_y,pos_z\n"

type grenadeTrajectoryRow struct {
	id           RowIndex
	grenadeId    RowIndex
	idPerGrenade int
	posX         float64
	posY         float64
	posZ         float64
}

func (g grenadeTrajectoryRow) toString() string {
	return fmt.Sprintf("%d,%d,%d,%.2f,%.2f,%.2f\n",
		g.id, g.grenadeId, g.idPerGrenade, g.posX, g.posY, g.posZ)
}

var grenadeTrajectoryTable table[grenadeTrajectoryRow]

// PLAYERFLASHED TABLE

const playerFlashedHeader = "id,tick_id,grenade_id,thrower,victim\n"

type playerFlashedRow struct {
	id        RowIndex
	tickId    RowIndex
	grenadeId RowIndex
	thrower   RowIndex
	victim    RowIndex
}

func (p playerFlashedRow) toString() string {
	return fmt.Sprintf("%d,%d,%d,%d,%d\n",
		p.id, p.tickId, p.grenadeId, p.thrower, p.victim)
}

var playerFlashedTable table[playerFlashedRow]

// PLANT TABLE

const plantHeader = "id,start_tick,end_tick,planter,successful\n"

type plantRow struct {
	//will reset to not valid at end of round
	id         RowIndex
	startTick  RowIndex
	endTick    RowIndex
	planter    RowIndex
	successful bool
}

func (p plantRow) toString() string {
	return fmt.Sprintf("%d,%d,%d,%d,%d\n",
		p.id, p.startTick, p.endTick, p.planter, boolToInt(p.successful))
}

var plantTable table[plantRow]

// DEFUSAL TABLE

const defusalHeader = "id,plant_id,start_tick,end_tick,defuser,successful\n"

type defusalRow struct {
	id         RowIndex
	plantId    RowIndex
	startTick  RowIndex
	endTick    RowIndex
	defuser    RowIndex
	successful bool
}

func (d defusalRow) toString() string {
	return fmt.Sprintf("%d,%d,%d,%d,%d,%d\n",
		d.id, d.plantId, d.startTick, d.endTick, d.defuser, boolToInt(d.successful))
}

var defusalTable table[defusalRow]

// EXPLOSION TABLE

const explosionHeader = "id,plant_id,tick_id\n"

type explosionRow struct {
	id      RowIndex
	plantId RowIndex
	tickId  RowIndex
}

func (e explosionRow) toString() string {
	return fmt.Sprintf("%d,%d,%d\n", e.id, e.plantId, e.tickId)
}

var explosionTable table[explosionRow]
