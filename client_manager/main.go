package main

import (
	"fmt"
    "github.com/fsnotify/fsnotify"
    "io"
    "io/ioutil"
    "log"
    "net/http"
    "os"
    "strings"
)

func main() {
    data, err := ioutil.ReadFile("config.txt")
    if err != nil {
        fmt.Println("config.txt file reading error: ", err)
        return
    }

    configs := strings.Split(string(data), "\n")
    downloadDir := configs[0]
    csgoConfigDir := configs[1]
    cfgSrc := downloadDir + string(os.PathSeparator) + "csknow.cfg"
    cfgDst := csgoConfigDir + string(os.PathSeparator) + "csknow.cfg"
    fmt.Println("CSGO config source: ", cfgSrc)
    fmt.Println("CSGO config destination: ", cfgDst)

    watcher, err := fsnotify.NewWatcher()
    if err != nil {
        log.Fatal(err)
    }
    defer watcher.Close()

    done := make(chan bool)
    go func() {
        for {
            select {
            case event, ok := <-watcher.Events:
                if !ok {
                    return
                }
                if ((event.Op&fsnotify.Write == fsnotify.Write) || (event.Op&fsnotify.Create == fsnotify.Create)) &&
                    (event.Name == cfgSrc){
                    log.Println("modified file:", event.Name)
                    copy(cfgSrc, cfgDst)
                }
            case err, ok := <-watcher.Errors:
                if !ok {
                    return
                }
                log.Println("error:", err)
            }
        }
    }()

    err = watcher.Add(downloadDir)
    if err != nil {
        log.Fatal(err)
    }
    <-done
}

// https://stackoverflow.com/a/21061062
// Copy the src file to dst. Any existing file will be overwritten and will not
// copy file attributes.
func copy(src string, dst string) error {
    in, err := os.Open(src)
    if err != nil {
        return err
    }
    defer in.Close()

    out, err := os.Create(dst)
    if err != nil {
        return err
    }
    defer out.Close()

    _, err = io.Copy(out, in)
    if err != nil {
        return err
    }
    return out.Close()
}

// https://golangcode.com/download-a-file-from-a-url/
// DownloadFile will download a url to a local file. It's efficient because it will
// write as it downloads and not load the whole file into memory.
func DownloadFile(filepath string, url string) error {

    // Get the data
    resp, err := http.Get(url)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    // Create the file
    out, err := os.Create(filepath)
    if err != nil {
        return err
    }
    defer out.Close()

    // Write the body to file
    _, err = io.Copy(out, resp.Body)
    return err
}