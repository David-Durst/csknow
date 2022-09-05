package constants

type GameType int

const (
	Pro GameType = iota
	Bot
	NUM_GAME_TYPES
)

var gameTypes = []string{"pros", "bots"}

func GameTypes() []string {
	return gameTypes
}

func GameTypeToString(gameType GameType) string {
	return gameTypes[gameType]
}

func GameTypeIntToString(gameType int) string {
	return gameTypes[gameType]
}
