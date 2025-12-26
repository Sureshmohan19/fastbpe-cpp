#!/usr/bin/env bash
set -euo pipefail

BPE=./bpe
CORPUS=docs/tinyshakespeare.txt
MODEL=model.bin
TMP=tmp_bpe_test

mkdir -p $TMP

echo "BPE COMPREHENSIVE TEST SUITE"
echo "-----------------------------"

# 1. Train tokenizer
echo "[1] Training tokenizer..."
$BPE train "$CORPUS" "$MODEL" 5000 1
echo "✓ Training completed"

# 2. Basic encode/decode round trip
echo "[2] Encode/decode round-trip test..."

TEXT="To be, or not to be: that is the question."
IDS=$($BPE encode "$MODEL" "$TEXT")
OUT=$($BPE decode "$MODEL" $IDS)

if [[ "$OUT" != "$TEXT" ]]; then
    echo "✗ Round-trip failed"
    echo "Expected: $TEXT"
    echo "Got:      $OUT"
    exit 1
fi
echo "✓ Round-trip passed"

# 3. Determinism test
echo "[3] Determinism test..."

IDS1=$($BPE encode "$MODEL" "$TEXT")
IDS2=$($BPE encode "$MODEL" "$TEXT")

if [[ "$IDS1" != "$IDS2" ]]; then
    echo "✗ Non-deterministic encode"
    exit 1
fi
echo "✓ Deterministic encoding"

# 4. Empty string
echo "[4] Empty string test..."

IDS_EMPTY=$($BPE encode "$MODEL" "")
OUT_EMPTY=$($BPE decode "$MODEL" $IDS_EMPTY)

if [[ "$OUT_EMPTY" != "" ]]; then
    echo "✗ Empty string failed"
    exit 1
fi
echo "✓ Empty string handled correctly"

# 5. Whitespace & punctuation stress
echo "[5] Whitespace & punctuation test..."

TEXT2="   !!! ???   \n\t  foo   bar!!!   "
IDS=$($BPE encode "$MODEL" "$TEXT2")
OUT=$($BPE decode "$MODEL" $IDS)

if [[ "$OUT" != "$TEXT2" ]]; then
    echo "✗ Whitespace/punctuation mismatch"
    exit 1
fi
echo "✓ Whitespace & punctuation preserved"

# 6. ASCII exhaustiveness (0–127)
echo "[6] ASCII exhaustiveness test..."

ASCII_TEXT=$(printf "%b" "$(printf '\\x%02x' {0..127})")
IDS=$($BPE encode "$MODEL" "$ASCII_TEXT")
OUT=$($BPE decode "$MODEL" $IDS)

if [[ "$OUT" != "$ASCII_TEXT" ]]; then
    echo "✗ ASCII test failed"
    exit 1
fi
echo "✓ ASCII round-trip OK"

# 7. Large input stability
echo "[7] Large input test..."

head -c 100000 "$CORPUS" > $TMP/large.txt
TEXT=$(cat $TMP/large.txt)
IDS=$($BPE encode "$MODEL" "$TEXT")
OUT=$($BPE decode "$MODEL" $IDS)

if [[ "$OUT" != "$TEXT" ]]; then
    echo "✗ Large input round-trip failed"
    exit 1
fi
echo "✓ Large input handled correctly"

# 8. Corrupted model detection
echo "[8] Corrupted model detection..."

cp "$MODEL" $TMP/bad.bin
printf "\x00\x00\x00\x00" | dd of=$TMP/bad.bin bs=1 seek=0 count=4 conv=notrunc &>/dev/null

if $BPE encode $TMP/bad.bin "test" &>/dev/null; then
    echo "✗ Corrupted model not detected"
    exit 1
fi
echo "✓ Corrupted model correctly rejected"

# 9. Re-load determinism
echo "[9] Reload determinism..."

IDS1=$($BPE encode "$MODEL" "$TEXT")
IDS2=$($BPE encode "$MODEL" "$TEXT")

if [[ "$IDS1" != "$IDS2" ]]; then
    echo "✗ Reload determinism failed"
    exit 1
fi
echo "✓ Reload determinism OK"

echo "ALL TESTS PASSED"
echo "----------------"
