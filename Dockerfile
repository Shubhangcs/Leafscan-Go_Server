# Use Go 1.23.5 builder image
FROM golang:1.23.5 AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -o server .

# Use base image with newer GLIBC (bookworm)
FROM debian:bookworm-slim

WORKDIR /app

COPY --from=builder /app/server .
COPY --from=builder /app/leafscan-d0ee4-firebase-adminsdk-fbsvc-cb14153170.json .

EXPOSE 9000

CMD ["./server"]
