#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <minutes> <command...>"
    echo "  Example: $0 10 echo hello"
    exit 1
fi

DELAY_MIN=$1
shift
CMD="$@"

DELAY=$((DELAY_MIN * 60))

START_TIME=$(date +%s)
EXEC_TIME=$((START_TIME + DELAY))
EXEC_TIME_STR=$(date -r $EXEC_TIME "+%Y-%m-%d %H:%M:%S" 2>/dev/null || date -d "@$EXEC_TIME" "+%Y-%m-%d %H:%M:%S")

echo "Command    : $CMD"
echo "Scheduled  : $EXEC_TIME_STR"
echo "Delay      : ${DELAY_MIN} min (${DELAY} sec)"
echo "----------------------------------------"

while true; do
    NOW=$(date +%s)
    REMAINING=$((EXEC_TIME - NOW))

    if [ $REMAINING -le 0 ]; then
        printf "\r\033[KScheduled: %s  |  Remaining: 0s  \n" "$EXEC_TIME_STR"
        break
    fi

    HOURS=$((REMAINING / 3600))
    MINUTES=$(( (REMAINING % 3600) / 60 ))
    SECONDS=$((REMAINING % 60))

    if [ $HOURS -gt 0 ]; then
        REMAINING_STR=$(printf "%02dh %02dm %02ds" $HOURS $MINUTES $SECONDS)
    elif [ $MINUTES -gt 0 ]; then
        REMAINING_STR=$(printf "%02dm %02ds" $MINUTES $SECONDS)
    else
        REMAINING_STR=$(printf "%02ds" $SECONDS)
    fi

    printf "\r\033[KScheduled: %s  |  Remaining: %s" "$EXEC_TIME_STR" "$REMAINING_STR"

    sleep 1
done

echo "▶ Running: $CMD"
eval "$CMD"
