package main

import (
    "bufio"
    "fmt"
    "github.com/fsnotify/fsnotify"
    "io"
    "io/ioutil"
    "log"
    "net/http"
    "os"
    "path/filepath"
    "strings"
    "time"
)

func main() {
    data, err := ioutil.ReadFile("config.txt")
    if err != nil {
        fmt.Println("config.txt file reading error: ", err)
        return
    }

    configs := strings.Split(string(data), "\n")
    downloadDir := strings.TrimRight(configs[0], "\r\n")
    csgoConfigDir := strings.TrimRight(configs[1], "\r\n")
    cfgSrc := downloadDir + string(os.PathSeparator) + "csknow.cfg"
	cfgSrcPrefix := downloadDir + string(os.PathSeparator) + "csknow"
    cfgDst := csgoConfigDir + string(os.PathSeparator) + "csknow.cfg"
    demoDstDir := filepath.Dir(csgoConfigDir) + string(os.PathSeparator)
    fmt.Println("CSGO config source: ", cfgSrc)
    fmt.Println("CSGO config destination: ", cfgDst)
    fmt.Println("CSGO demo destination folder: ", demoDstDir)

    watcher, err := fsnotify.NewWatcher()
    if err != nil {
        log.Fatal(err)
    }
    defer watcher.Close()

    done := make(chan bool)
    nextCfgFileDownloaded := ""
    lastCfgFileDownloaded := ""
    go func() {
        for (true) {
        	if nextCfgFileDownloaded != lastCfgFileDownloaded {
        	    lastCfgFileDownloaded = nextCfgFileDownloaded
                downloadDemo(lastCfgFileDownloaded, demoDstDir)
                copy(lastCfgFileDownloaded, cfgDst)
                log.Println("downloaded demo")
            }
            time.Sleep(5 * time.Second)
        }
    }()
    go func() {
        for {
            select {
            case event, ok := <-watcher.Events:
                if !ok {
                    return
                }
                if ((event.Op&fsnotify.Write == fsnotify.Write) || (event.Op&fsnotify.Create == fsnotify.Create)) &&
                    strings.Contains(event.Name, cfgSrcPrefix)  {
                	log.Println("event string: ", event.String())
                	nextCfgFileDownloaded = event.Name
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

func downloadDemo(filepath string, dst string) {
    cfgFile, err := os.OpenFile(filepath, os.O_APPEND|os.O_CREATE|os.O_RDONLY, 0644)
    if err != nil {
        panic(err)
    }
    defer cfgFile.Close()

    reader := bufio.NewReader(cfgFile)
    demoURL, _, _ := reader.ReadLine()
    demoName, _, _ := reader.ReadLine()
    downloadFile(dst, trimLeftChars(string(demoURL), 2), trimLeftChars(string(demoName), 2))
}

// https://golangcode.com/download-a-file-from-a-url/
// DownloadFile will download a url to a local file. It's efficient because it will
// write as it downloads and not load the whole file into memory.
func downloadFile(filepath string, url string, demoName string) error {

    // Get the data
    resp, err := http.Get(url)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    // Create the file
    out, err := os.Create(filepath + demoName)
    if err != nil {
        return err
    }
    defer out.Close()

    // Write the body to file
    _, err = io.Copy(out, resp.Body)
    return err
}

func trimLeftChars(s string, n int) string {
    m := 0
    for i := range s {
        if m >= n {
            return s[i:]
        }
        m++
    }
    return s[:0]
}