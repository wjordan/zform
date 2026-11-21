#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${1:?pass a corpus directory}"
cd "$1"

has_hash() {
  [[ -f $2 ]] || return 1
  printf '%s  %s\n' "$1" "$2" | sha256sum --check --status
}

download() {
  local url=$1 hash=$2 path=$3
  if ! has_hash "$hash" "$path" 2>/dev/null; then
    curl --fail --location --silent --show-error --retry 3 --max-time 480 "$url" -o "$path.part"
    has_hash "$hash" "$path.part"
    mv "$path.part" "$path"
  fi
}

extract() {
  local archive=$1 member=$2 hash=$3
  if ! has_hash "$hash" "$member" 2>/dev/null; then
    unzip -p "$archive" "$member" > "$member.part"
    has_hash "$hash" "$member.part"
    mv "$member.part" "$member"
  fi
}

download https://zenodo.org/records/8196385/files/HDFS_v1.zip \
  04f919f2185821f23f045dca611a7586429bdabc601bd7b43f30005f8e289b01 HDFS_v1.zip
extract HDFS_v1.zip HDFS.log \
  0783096174d7832c618337f9609e06e04abd86ddd7089b3c12b407e63bfebc52

download https://data.gharchive.org/2020-06-17-16.json.gz \
  c7082d5c661b2e3298cb40e57ff55120926ed3762d4f1e8c7caf7b73e74481c9 2020-06-17-16.json.gz

download https://mattmahoney.net/dc/silesia.zip \
  7d1dd71bfecda66a0ca30d863ed031809f67ecf12717a60fe72c1cc39e28434e silesia.zip
extract silesia.zip sao \
  c2d0ea2cc59d4c21b7fe43a71499342a00cbe530a1d5548770e91ecd6214adcc
