package internal

import (
	"fmt"
	c "github.com/David-Durst/csknow/demo_parser/internal/constants"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/events"
	"github.com/oklog/ulid/v2"
	"os"
	"path/filepath"
	"strings"
)

type RowIndex int64

const InvalidId RowIndex = -1
const InvalidInt int = -1
const InvalidFloat float32 = -1.

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
	nextSay               RowIndex
}

func DefaultIDState() IDState {
	return IDState{0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0,
		0, 0, 0}
}

type tableRow interface {
	ID() RowIndex
	toString() string
}

type table[T tableRow] struct {
	rows       []T
	firstRowID RowIndex
	file       *os.File
}

func (t *table[T]) init(fileName string, header string) {
	t.rows = make([]T, 0)
	file, err := os.Create(filepath.Join(c.TmpDir, fileName))
	if err != nil {
		panic(err)
	}
	t.file = file
	t.file.WriteString(header)
}

func (t *table[T]) remove(fileName string) {
	err := os.Remove(filepath.Join(c.TmpDir, fileName))
	if err != nil {
		panic(err)
	}
}

func (t *table[T]) tail() *T {
	return &t.rows[len(t.rows)-1]
}

func (t *table[T]) get(i RowIndex) *T {
	return &t.rows[i-t.firstRowID]
}

func (t *table[T]) append(e T) {
	if len(t.rows) == 0 {
		t.firstRowID = e.ID()
	}

	t.rows = append(t.rows, e)
}

func (t *table[T]) len() int {
	return len(t.rows)
}

func (t *table[T]) flush(close bool) {
	var sb strings.Builder
	// save last row if not closing
	var lastRow T
	savedValue := false
	if t.len() > 0 {
		lastRow = *t.tail()
		savedValue = true
	}
	for i, element := range t.rows {
		if close || i < len(t.rows)-1 {
			sb.WriteString(element.toString())
		}
	}
	t.file.WriteString(sb.String())
	t.rows = make([]T, 0)
	if close {
		t.file.Close()
	} else if savedValue {
		t.append(lastRow)
	}
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

func (p playerRow) ID() RowIndex {
	return p.id
}

var playersTable table[playerRow]

type playersTrackerT struct {
	gameIdToTableId map[int]RowIndex
}

func (p *playersTrackerT) init() {
	p.gameIdToTableId = make(map[int]RowIndex)
}

func (p *playersTrackerT) addPlayer(pr playerRow, player *common.Player) {
	p.gameIdToTableId[player.UserID] = pr.id
	playersTable.append(pr)
}

func (p *playersTrackerT) alreadyAddedPlayer(player *common.Player) bool {
	_, ok := p.gameIdToTableId[player.UserID]
	return ok
}

func (p *playersTrackerT) getPlayerIdFromGameData(player *common.Player) RowIndex {
	if player == nil {
		return InvalidId
	}

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

func (r roundRow) ID() RowIndex {
	return r.id
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

func (t tickRow) ID() RowIndex {
	return t.id
}

var ticksTable table[tickRow]

// PLAYERATTICKS TABLE

const playerAtTicksHeader = "id,player_id,tick_id,pos_x,pos_y,pos_z,eye_pos_z,vel_x,vel_y,vel_z,view_x,view_y,aim_punch_x,aim_punch_y," +
	"view_punch_x,view_punch_y,recoilIndex,next_primary_attack,next_secondary_attack,game_time,team,health,armor,has_helmet,is_alive,ducking_key_pressed,duck_amount,is_reloading,is_walking,is_scoped," +
	"is_airborne,flash_duration,active_weapon,main_weapon,primary_bullets_clip," +
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
	recoilIndex             float32
	nextPrimaryAttack       float32
	nextSecondaryAttack     float32
	gameTime                float64
	team                    int
	health                  int
	armor                   int
	hasHelmet               bool
	isAlive                 bool
	duckingKeyPressed       bool
	duckAmount              float32
	isReloading             bool
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
			"%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.8f,%.8f,%.8f,"+
			"%d,%d,%d,%d,"+
			"%d,%d,%.2f,%d,%d,%d,%d,"+
			"%f,%d,%d,%d,"+
			"%d,%d,%d,%d,"+
			"%d,%d,%d,%d,%d,%d,%d,"+
			"%d,%d,%d,%d\n",
		p.id, p.playerId, p.tickId, p.posX, p.posY, p.posZ, p.eyePosZ,
		p.velX, p.velY, p.velZ,
		p.viewX, p.viewY, p.aimPunchX, p.aimPunchY, p.viewPunchX, p.viewPunchY, p.recoilIndex, p.nextPrimaryAttack, p.nextSecondaryAttack, p.gameTime,
		p.team, p.health, p.armor, boolToInt(p.hasHelmet),
		boolToInt(p.isAlive), boolToInt(p.duckingKeyPressed), p.duckAmount, boolToInt(p.isReloading), boolToInt(p.isWalking), boolToInt(p.isScoped), boolToInt(p.isAirborne),
		p.flashDuration, p.activeWeapon, p.primaryWeapon, p.primaryBulletsClip,
		p.primaryBulletsReserve, p.secondaryWeapon, p.secondaryBulletsClip, p.secondaryBulletsReserve,
		p.numHE, p.numFlash, p.numSmoke, p.numIncendiary, p.numMolotov, p.numDecoy, p.numZeus,
		boolToInt(p.hasDefuser), boolToInt(p.hasBomb), p.money, p.ping)
}

func (p playerAtTickRow) ID() RowIndex {
	return p.id
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

func (s spottedRow) ID() RowIndex {
	return s.id
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

func (f footstepRow) ID() RowIndex {
	return f.id
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

func (w weaponFireRow) ID() RowIndex {
	return w.id
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

func (h hurtRow) ID() RowIndex {
	return h.id
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

func (k killRow) ID() RowIndex {
	return k.id
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
}

func (g grenadeRow) toString() string {
	return fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d\n",
		g.id, g.thrower, g.grenadeType, g.throwTick, g.activeTick, g.expiredTick, g.destroyTick)
}

func (g grenadeRow) ID() RowIndex {
	return g.id
}

var grenadeTable table[grenadeRow]

type grenadeTrackerT struct {
	uniqueIdToTableId map[ulid.ULID]RowIndex
}

func (g *grenadeTrackerT) init() {
	g.uniqueIdToTableId = make(map[ulid.ULID]RowIndex)
}

func (g *grenadeTrackerT) addGrenade(gr grenadeRow, grenade *common.Equipment) {
	g.uniqueIdToTableId[grenade.UniqueID2()] = gr.id
	grenadeTable.append(gr)
}

func (g *grenadeTrackerT) alreadyAddedGrenade(grenade *common.Equipment) bool {
	// smoke grenades can be nil (319_titan-epsilon_de_dust2.dem)
	if grenade == nil {
		return false
	}
	_, ok := g.uniqueIdToTableId[grenade.UniqueID2()]
	return ok
}

func (g *grenadeTrackerT) getGrenadeIdFromGameData(grenade *common.Equipment) RowIndex {
	if tableId, ok := g.uniqueIdToTableId[grenade.UniqueID2()]; ok {
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

func (g grenadeTrajectoryRow) ID() RowIndex {
	return g.id
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

func (p playerFlashedRow) ID() RowIndex {
	return p.id
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

func (p plantRow) ID() RowIndex {
	return p.id
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

func (d defusalRow) ID() RowIndex {
	return d.id
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

func (e explosionRow) ID() RowIndex {
	return e.id
}

var explosionTable table[explosionRow]

// SAY TABLE

const sayHeader = "id,tick_id,message\n"

type sayRow struct {
	id      RowIndex
	tickId  RowIndex
	message string
}

func (e sayRow) toString() string {
	return fmt.Sprintf("%d,%d,%s\n", e.id, e.tickId, e.message)
}

func (e sayRow) ID() RowIndex {
	return e.id
}

var sayTable table[sayRow]
