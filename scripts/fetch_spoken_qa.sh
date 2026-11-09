#!/usr/bin/env bash
# Download real spoken questions (audio, to the HDD) and real-user chat feedback
# (text, to the NVMe), raw parquet/json as published. Existing files are skipped;
# an interrupted download resumes from its .part.
#
#   heysquad_human  fixie-ai/HeySQuAD_human: 76k SQuAD questions read by people
#   voicebench      hlt-lab/voicebench, human-recorded subsets only: wildvoice
#                   (open questions, many accents), commoneval, sd-qa (11 dialects)
#   wildspeech      tencent/WildSpeech-Bench: 1100 real WildChat queries; the 100
#                   paralinguistic ones are recorded, the rest are CosyVoice clones
#   wildfeedback    microsoft/WildFeedback: WildChat with per-turn SAT/DSAT labels
set -u
AUDIO=${AUDIO:-/run/media/unschlagbar/BTRFS/training_data}
TEXT=${TEXT:-$HOME/training_data}
HF=https://huggingface.co/datasets

fetch() { # url out
    [ -s "$2" ] && return 0
    mkdir -p "$(dirname "$2")"
    curl -sfL --retry 20 --retry-delay 15 -C - -o "$2.part" "$1" && mv "$2.part" "$2" && echo "ok   $2" || echo "FAIL $2"
}

files() { # repo prefix
    curl -s "https://huggingface.co/api/datasets/$1/tree/main/$2?recursive=1" |
        python3 -I -c "import json,sys;[print(f['path']) for f in json.load(sys.stdin) if f['type']=='file' and f['path'].endswith('.parquet')]"
}

list=()
for f in $(files fixie-ai/HeySQuAD_human data); do
    list+=("$HF/fixie-ai/HeySQuAD_human/resolve/main/$f $AUDIO/heysquad_human/$f")
done
for sub in wildvoice commoneval sd-qa; do
    for f in $(files hlt-lab/voicebench "$sub"); do
        list+=("$HF/hlt-lab/voicebench/resolve/main/$f $AUDIO/voicebench/$f")
    done
done
list+=("$HF/tencent/WildSpeech-Bench/resolve/main/data/wildspeech_1100.parquet $AUDIO/wildspeech/wildspeech_1100.parquet")
list+=("$HF/tencent/WildSpeech-Bench/resolve/main/License.txt $AUDIO/wildspeech/License.txt")
for f in wildfeedback.json sat_dsat_annotation.json user_preference.json; do
    list+=("$HF/microsoft/WildFeedback/resolve/main/$f $TEXT/wildfeedback/$f")
done

for d in heysquad_human voicebench wildspeech; do
    mkdir -p "$AUDIO/$d"
    [ -e "$TEXT/$d" ] || ln -s "$AUDIO/$d" "$TEXT/$d"
done

export -f fetch
printf '%s\n' "${list[@]}" | xargs -P 6 -L 1 bash -c 'fetch "$0" "$1"'
