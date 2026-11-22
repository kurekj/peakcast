#!/bin/bash
# Live experiments on two GPUs (run after make_calibration.py has written results/lofo).
#   bash run_live.sh PYTHON OUTPUT_DIRECTORY CALIBRATION_DIRECTORY
set -u
PY=$1
OUT=$2
CAL=$3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/compare" "$OUT/buckets" "$OUT/protocol"

COMPARE="resnet50 vgg16 densenet121 mobilenet_v2 efficientnet_b0 convnext_tiny regnet_y_1_6gf vit_b_16 swin_t
inception_v3 timm_deit_small_patch16_224 timm_mixer_b16_224 timm_hrnet_w18 timm_pvt_v2_b2 fcn_resnet50
custom_unet_32 r3d_18 hf_bert_base hf_distilbert hf_roberta_base hf_electra_small hf_gpt2_small hf_gpt2_medium
hf_llama_m hf_pythia_160m hf_t5_small hf_bart_base hf_wav2vec2_base hf_whisper_base custom_lstm_lm
custom_torch_encoder_lm hf_clip_b32"
BUCKETS="hf_bert_base hf_distilbert hf_roberta_base hf_bert_large hf_gpt2_small hf_gpt2_medium hf_llama_m
hf_pythia_160m hf_t5_small hf_bart_base custom_lstm_lm custom_torch_encoder_lm"
PROTOCOL="resnet50 vgg16 densenet121 efficientnet_b0 convnext_tiny vit_b_16 swin_t hf_bert_base hf_gpt2_small
hf_t5_small hf_llama_m custom_lstm_lm"

worker() {
  gpu=$1
  shift
  for job in "$@"; do
    kind=${job%%:*}
    model=${job#*:}
    case $kind in
      compare)
        [ -s "$OUT/compare/$model.jsonl" ] && continue
        CUDA_VISIBLE_DEVICES=$gpu timeout 3000 "$PY" "$HERE/live_compare.py" --model "$model" \
          --out "$OUT/compare/$model.jsonl" --caps "${CAPS_COMPARE:-8,24,full}" --risks "${RISKS:-0.01,0.05}" --calibration "$CAL" \
          > "$OUT/compare/$model.log" 2>&1 ;;
      buckets)
        [ -s "$OUT/buckets/$model.jsonl" ] && continue
        CUDA_VISIBLE_DEVICES=$gpu timeout 1800 "$PY" "$HERE/live_buckets.py" --model "$model" \
          --out "$OUT/buckets/$model.jsonl" --caps "${CAPS_BUCKETS:-8,16}" --risk "${BUCKET_RISK:-0.05}" --calibration "$CAL" \
          > "$OUT/buckets/$model.log" 2>&1 ;;
      protocol)
        for b in 8 64; do
          CUDA_VISIBLE_DEVICES=$gpu timeout 600 "$PY" "$HERE/validate_protocol.py" --model "$model" \
            --batch $b --out "$OUT/protocol/gpu$gpu.jsonl" > /dev/null 2>&1
        done ;;
    esac
    echo "$(date +%T) gpu$gpu $job" >> "$OUT/_live.log"
  done
}

jobs0=()
jobs1=()
i=0
for m in $PROTOCOL; do j="protocol:$m"; if [ $((i % 2)) = 0 ]; then jobs0+=("$j"); else jobs1+=("$j"); fi; i=$((i + 1)); done
for m in $BUCKETS; do j="buckets:$m"; if [ $((i % 2)) = 0 ]; then jobs0+=("$j"); else jobs1+=("$j"); fi; i=$((i + 1)); done
for m in $COMPARE; do j="compare:$m"; if [ $((i % 2)) = 0 ]; then jobs0+=("$j"); else jobs1+=("$j"); fi; i=$((i + 1)); done
if [ "${GPUS:-2}" = 1 ]; then
  worker 0 "${jobs0[@]}" "${jobs1[@]}"
else
  worker 0 "${jobs0[@]}" &
  worker 1 "${jobs1[@]}" &
  wait
fi
echo LIVE_DONE >> "$OUT/_live.log"
