package main

import (
    "fmt"
    "github.com/galaco/bsp"
    "github.com/galaco/bsp/lumps"
    "log"
    "os"
)

func main() {

    r, err := bsp.ReadFromFile("maps/v20/de_dust2.bsp")
    if err != nil {
        fmt.Println(err)
    }


    lump := r.Lump(bsp.LumpEntities).(*lumps.Entities)
    log.Println(lump.GetData())
}
