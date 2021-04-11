mkdir -p ../demos
docker run --name durst_pro_demos_downloader \
    --rm -it durst/csgo-pro-demos-downloader:0.1 /bin/bash
