FROM golang:1.16

RUN apt update && \
    apt install -y vim neovim gdb && \
    echo "set auto-load safe-path /" >> /root/.gdbinit

WORKDIR /go/src/app
COPY . .

RUN go build .

CMD ["go", "run", "."]
