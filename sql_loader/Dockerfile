FROM mariadb:10.5.9-focal

RUN apt update && \
    apt install -y vim neovim gdb && \
    echo "set auto-load safe-path /" >> /root/.gdbinit

COPY . .
