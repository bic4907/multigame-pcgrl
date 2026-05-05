#!/usr/bin/env bash
set -euo pipefail

DEFAULT_SSH_COMMAND="ssh root@213.173.105.96 -p 46692 -i ~/.ssh/id_ed25519"
DEFAULT_REMOTE="/workspace/nas/mgpcgrl/"
DEFAULT_DEST="/mnt/nas/mgpcgrl"

read -r -p "SSH command [${DEFAULT_SSH_COMMAND}]: " SSH_COMMAND
SSH_COMMAND="${SSH_COMMAND:-$DEFAULT_SSH_COMMAND}"

read -r -p "Remote path [${DEFAULT_REMOTE}]: " REMOTE_PATH
REMOTE_PATH="${REMOTE_PATH:-$DEFAULT_REMOTE}"
REMOTE_SOURCE="${REMOTE_PATH%/}/"

read -r -p "Destination path [${DEFAULT_DEST}]: " DEST_PATH
DEST_PATH="${DEST_PATH:-$DEFAULT_DEST}"
if [ "$DEST_PATH" = '$(pwd)' ]; then
  DEST_PATH="$PWD"
fi

read -r -a SSH_TOKENS <<< "$SSH_COMMAND"
if [ "${#SSH_TOKENS[@]}" -eq 0 ] || [ "${SSH_TOKENS[0]}" != "ssh" ]; then
  echo "Error: SSH command must start with 'ssh'"
  exit 1
fi

SSH_TARGET=""
SSH_RSH_TOKENS=("ssh")
i=1
while [ "$i" -lt "${#SSH_TOKENS[@]}" ]; do
  token="${SSH_TOKENS[$i]}"

  if [ "${token#-}" != "$token" ]; then
    case "$token" in
      -4|-6|-A|-a|-C|-f|-G|-g|-K|-k|-M|-N|-n|-q|-s|-T|-t|-V|-v|-X|-x|-Y|-y)
        SSH_RSH_TOKENS+=("$token")
        ;;
      -B|-b|-c|-D|-E|-e|-F|-I|-i|-J|-L|-l|-m|-O|-o|-p|-Q|-R|-S|-W|-w)
        SSH_RSH_TOKENS+=("$token")
        i=$((i + 1))
        if [ "$i" -lt "${#SSH_TOKENS[@]}" ]; then
          SSH_RSH_TOKENS+=("${SSH_TOKENS[$i]}")
        fi
        ;;
      *)
        SSH_RSH_TOKENS+=("$token")
        ;;
    esac
  elif [ -z "$SSH_TARGET" ]; then
    SSH_TARGET="$token"
  else
    SSH_RSH_TOKENS+=("$token")
  fi

  i=$((i + 1))
done

if [ -z "$SSH_TARGET" ]; then
  echo "Error: could not parse user@host from SSH command"
  exit 1
fi

printf -v SSH_RSH_COMMAND "%q " "${SSH_RSH_TOKENS[@]}"
SSH_RSH_COMMAND="${SSH_RSH_COMMAND% }"

mkdir -p "$DEST_PATH"

APPEND_FLAG="--append"
if rsync --help 2>/dev/null | grep -q -- "--append-verify"; then
  APPEND_FLAG="--append-verify"
fi

rsync -avhP \
  --partial \
  "$APPEND_FLAG" \
  -e "$SSH_RSH_COMMAND" \
  "${SSH_TARGET}:${REMOTE_SOURCE}" \
  "$DEST_PATH"
