FROM golang:1.14

RUN apt-get -y update && apt-get -y install less procps unzip zip python3

WORKDIR /go/src/app
COPY . .

CMD ["python3"]