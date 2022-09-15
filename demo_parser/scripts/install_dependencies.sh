sudo apt install jq
export GOENV_LATEST_VERSION=$(curl --silent "https://api.github.com/repos/spacewalkio/Goenv/releases/latest" | jq '.tag_name' | sed -E 's/.*"([^"]+)".*/\1/' | tr -d v)
mkdir goenv
cd goenv
curl -sL https://github.com/spacewalkio/Goenv/releases/download/v{$GOENV_LATEST_VERSION}/goenv_{$GOENV_LATEST_VERSION}_Linux_x86_64.tar.gz | tar xz
cd ..
sudo rm -rf /opt/goenv
sudo mv goenv /opt/
sudo chmod a+x /opt/goenv/goenv
/opt/goenv/goenv config
/opt/goenv/goenv install 1.19
/opt/goenv/goenv global 1.19

if [[ -n "$1" ]] && [[ "$1" == "y" ]]; then
    echo -e "\n\n# Added By CSKnow" >> ~/.profile
    echo 'export PATH="$HOME/.goenv/shims:/opt/goenv"$PATH' >> ~/.profile
else
    echo add \"'export PATH="$HOME/.goenv/shims:/opt/goenv"$PATH'\" to ~/.profile
fi

