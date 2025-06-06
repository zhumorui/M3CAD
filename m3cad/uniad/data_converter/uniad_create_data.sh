
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py openv2v_4_cams --root-path ./data/openv2v_4_cams \
       --out-dir ./data/infos_mini \
       --extra-tag openv2v_4_cams \
       --version v1.0-mini \
       --canbus ./data/openv2v_4_cams \
