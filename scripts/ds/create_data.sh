export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python nuscenes/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0-mini

python nuscenes/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0
