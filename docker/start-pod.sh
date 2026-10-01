#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PUBLIC_KEY:-}" ]]; then
    echo "PUBLIC_KEY is required; refusing to start a Pod without SSH access." >&2
    exit 1
fi

install -d -m 0700 /root/.ssh
install -d -m 0755 /run/sshd
authorized_keys=/root/.ssh/authorized_keys
temporary_key=""
trap 'rm -f "${temporary_key:-}"' EXIT
umask 077
: > "${authorized_keys}"
key_count=0

while IFS= read -r public_key; do
    [[ -z "${public_key}" ]] && continue
    temporary_key="$(mktemp)"
    printf '%s\n' "${public_key}" > "${temporary_key}"
    if ! ssh-keygen -l -f "${temporary_key}" >/dev/null 2>&1; then
        echo "PUBLIC_KEY contains an invalid SSH public key." >&2
        exit 1
    fi
    printf '%s\n' "${public_key}" >> "${authorized_keys}"
    rm -f "${temporary_key}"
    temporary_key=""
    key_count=$((key_count + 1))
done < <(printf '%s\n' "${PUBLIC_KEY}")

if [[ "${key_count}" -eq 0 ]]; then
    echo "PUBLIC_KEY did not contain an SSH public key." >&2
    exit 1
fi

chmod 0600 "${authorized_keys}"
ssh-keygen -A
exec /usr/sbin/sshd -D -e
