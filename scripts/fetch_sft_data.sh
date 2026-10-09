#!/usr/bin/env bash
# Download the raw, unfiltered SFT corpora that mixes/assistant_full.toml reads.
# Filtering happens in datamix at build time, so this is only ever needed once.
# Existing files are skipped; an interrupted download resumes from its .part.
set -u
DEST=${DEST:-$HOME/training_data}
HF=https://huggingface.co/datasets

fetch() { # url out
    [ -s "$2" ] && return 0
    mkdir -p "$(dirname "$2")"
    curl -sfL --retry 10 -C - -o "$2.part" "$1" && mv "$2.part" "$2" && echo "ok   $2" || echo "FAIL $2"
}

list=()
for sub in everyday-conversations explore-instruct-rewriting metamathqa-50k numina-cot-100k \
           openhermes-100k smol-constraints smol-magpie-ultra smol-rewrite smol-summarize; do
    for f in $(curl -s "https://huggingface.co/api/datasets/HuggingFaceTB/smoltalk/tree/main/data/$sub" |
               python3 -c "import json,sys;[print(f['path']) for f in json.load(sys.stdin) if '/train-' in f['path']]"); do
        list+=("$HF/HuggingFaceTB/smoltalk/resolve/main/$f $DEST/smoltalk/$sub/$(basename "$f")")
    done
done
list+=("$HF/HuggingFaceH4/no_robots/resolve/main/data/train-00000-of-00001.parquet $DEST/no_robots/train.parquet")
for i in $(seq -w 0 13); do
    list+=("$HF/allenai/WildChat-1M/resolve/main/data/train-000$i-of-00014.parquet $DEST/wildchat/train-000$i-of-00014.parquet")
done
list+=("$HF/OpenAssistant/oasst2/resolve/main/2023-11-05_oasst2_ready.trees.jsonl.gz $DEST/oasst2/ready.trees.jsonl.gz")

# HF serves ~1 MB/s per stream, so fetch in parallel.
export -f fetch
printf '%s\n' "${list[@]}" | xargs -P 12 -L 1 bash -c 'fetch "$0" "$1"'

[ -s "$DEST/oasst2/oasst2.jsonl" ] ||
    python3 "$(dirname "$0")/oasst2_to_jsonl.py" "$DEST/oasst2/ready.trees.jsonl.gz" "$DEST/oasst2/oasst2.jsonl"
# Needs pyarrow: the flags it filters on are BOOLEAN columns (see the script).
[ -s "$DEST/wildchat/wildchat_gpt4.jsonl" ] ||
    python3 "$(dirname "$0")/wildchat_to_jsonl.py" "$DEST/wildchat" "$DEST/wildchat"
