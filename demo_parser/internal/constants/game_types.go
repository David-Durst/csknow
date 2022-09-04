package constants

type GameType int

const (
	Pro GameType = 0
	Bot
)

var gameTypes = []string{"pros", "bots"}

func GameTypeToString(gameType GameType) string {
	return gameTypes[gameType]
}
