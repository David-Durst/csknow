FROM mariadb/columnstore:latest

RUN apt update && \
    apt install -y vim neovim gdb && \
    echo "set auto-load safe-path /" >> /root/.gdbinit

COPY . .
