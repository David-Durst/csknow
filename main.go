package main

import (
	"fmt"
	"os"
	"sort"

	dem "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs"
	events "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/events"
)

func main() {
	f, err := os.Open("./test_2_17_21.dem")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	p := dem.NewParser(f)
	defer p.Close()

	// Register handler on kill events
	fmt.Printf("tick number,game phase,rounds played,num players")
	for i := 0; i < 10; i++ {
		fmt.Printf(",player %d name,player %d x postion,player %d y position,player %d z position,player %d x view direction,player %d y view direction",i,i,i,i,i,i)
	}
	fmt.Printf("\n")
	p.RegisterEventHandler(func(e events.FrameDone) {
		gs := p.GameState()
		fmt.Printf("%d,%d,%d,", p.CurrentFrame(), gs.GamePhase(), gs.TotalRoundsPlayed())
		players := gs.Participants().Playing()
		sort.Slice(players, func(i int, j int) bool {
			return players[i].Name < players[j].Name
		})
		for i := 0; i < 10; i++ {
			if i >= len(players) {
				fmt.Printf(",,,,,")
			} else {
				fmt.Printf("%s,%.2f,%.2f,%.2f,%.2f,%.2f", players[i].Name,
					players[i].Position().X,players[i].Position().Y, players[i].Position().Z,
					players[i].ViewDirectionX(), players[i].ViewDirectionY())
			}
			if i < 9 {
				fmt.Printf(",")
			}
		}
		fmt.Print("\n")
		/*
		cts := gs.TeamCounterTerrorists()
		for i := 0; i < len(cts.Members()); i++ {
			member := cts.Members()[i]
			fmt.Printf("Player %d position: %s\n", member.Name, cts.Members()[i].Position().String())
		}
		 */
	})

	// Parse to end
	err = p.ParseToEnd()
	if err != nil {
		panic(err)
	}
}
