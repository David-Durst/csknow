FROM golang:1.16

RUN apt update && \
    apt install -y vim neovim gdb

WORKDIR /go/src/app
COPY . .

CMD ["pwd"]
