FROM postgres:13.2

RUN apt update && \
    apt install -y vim neovim gdb && \
    echo "set auto-load safe-path /" >> /root/.gdbinit

COPY . .
