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
echo "### ladder alignment"
python3 scripts/check_ladder_alignment.py 2>&1 | tail -2
[ ${PIPESTATUS[0]} -ne 0 ] && FAIL=1
echo
[ $FAIL -eq 0 ] && echo "ALL SUITES GREEN" || echo "SOME SUITES FAILED"
exit $FAIL
