#!/bin/bash
# ghcr-tag.sh [--if-missing] SOURCE TAG
#
# Point TAG at the image SOURCE (a tag, or a sha256: digest) on GHCR. Only
# the manifest is copied, so nothing is pulled or pushed, and TAG gets the
# same digest -- and so the same cosign signature -- as SOURCE. With
# --if-missing a TAG that already exists is left alone. Prints the digest
# TAG ends up at.
#
# Needs REPO (owner/name), ACTOR and GITHUB_TOKEN with packages: write.
set -euo pipefail

if_missing=false
if [ "${1:-}" = --if-missing ]; then
    if_missing=true
    shift
fi
src=$1
tag=$2

accept='application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'
url="https://ghcr.io/v2/$REPO/manifests"
token=$(curl -fsS -u "$ACTOR:$GITHUB_TOKEN" \
    "https://ghcr.io/token?scope=repository:$REPO:pull,push" \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')
auth="Authorization: Bearer $token"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
header() { tr -d '\r' < "$tmp/headers" | awk -F': ' -v h="$1" 'tolower($1) == h {print $2}'; }

status=$(curl -sS -D "$tmp/headers" -o /dev/null -w '%{http_code}' \
    -H "$auth" -H "Accept: $accept" "$url/$tag")
case "$status" in
    200)
        if $if_missing; then
            echo "$tag already exists; left as it is" >&2
            header docker-content-digest
            exit 0
        fi
        ;;
    404) ;;
    *) echo "GHCR answered $status for $tag" >&2; exit 1 ;;
esac

curl -fsS -D "$tmp/headers" -o "$tmp/manifest" -H "$auth" -H "Accept: $accept" "$url/$src"
digest=$(header docker-content-digest)
# The bytes as GHCR served them, so the copy hashes to the same digest.
curl -fsS -X PUT -H "$auth" -H "Content-Type: $(header content-type)" \
    --data-binary "@$tmp/manifest" "$url/$tag" > /dev/null
echo "$tag now points at $src ($digest)" >&2
echo "$digest"
