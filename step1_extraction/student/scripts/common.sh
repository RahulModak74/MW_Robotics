# sourced by the run_*.sh / kimodo_data/*.sh scripts. Set PM to your ProtoMotions clone and PY to its Python (MuJoCo backend).
R=${R:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}            # this folder (step1_extraction/student)
PM=${PM:-$HOME/ProtoMotions}
PY=${PY:-python}
OURS=$R/ours
export PYTHONPATH="$PM:$OURS:$R/../flight_computer" TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CK=data/pretrained_models/motion_tracker/g1-bones-deploy/inference_last.ckpt   # relative to $PM (download it with ProtoMotions' instructions)
