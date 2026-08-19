#!/bin/bash
# Full regression suite. Run before any training launch.
cd "$(dirname "$0")/.."
FAIL=0
for t in test_p0_fixes test_p1_icnf test_p2_icnf_arch; do
  echo "### $t"
  python3 scripts/$t.py 2>&1 | tail -3
  [ ${PIPESTATUS[0]} -ne 0 ] && FAIL=1
  echo
done
if [ -d data/real_lowlight_probe ]; then
  echo "### test_p3b_real_images (real photographs)"
  python3 scripts/test_p3b_real_images.py --data data/real_lowlight_probe 2>&1 | tail -3
  [ ${PIPESTATUS[0]} -ne 0 ] && FAIL=1
  echo
fi
echo "### ladder alignment"
python3 scripts/check_ladder_alignment.py 2>&1 | tail -2
[ ${PIPESTATUS[0]} -ne 0 ] && FAIL=1
echo
[ $FAIL -eq 0 ] && echo "ALL SUITES GREEN" || echo "SOME SUITES FAILED"
exit $FAIL
