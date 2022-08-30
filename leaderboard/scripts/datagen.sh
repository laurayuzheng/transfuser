export CARLA_ROOT=${1:-/home/laura/DrivingSimulators/CARLA_0.9.10}
export WORK_DIR=${2:-/scratch/transfuser}

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# export SCENARIOS=${WORK_DIR}/leaderboard/data/training/scenarios/Scenario10/Town10HD_Scenario10.json
# export ROUTES=${WORK_DIR}/leaderboard/data/training/routes/Scenario10/Town10HD_Scenario10.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=MAP
# export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/Town10HD_Scenario10.json
# export SAVE_PATH=${WORK_DIR}/results/Town10HD_Scenario10
export TEAM_AGENT=${WORK_DIR}/team_code_autopilot/cosim_wrapper.py
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=1

SCENARIO_DIR="${WORK_DIR}/leaderboard/data/training/scenarios"
ROUTES_DIR="${WORK_DIR}/leaderboard/data/training/routes"

for scenario in "$SCENARIO_DIR"/*
do
#   scenario_basename="${scenario%.*}"
  scenario_basename=$(basename ${scenario%.*}) 
#   echo "Scenario basename: ${scenario_basename}"
  SCENARIO_ROUTES="${ROUTES_DIR}/$scenario_basename" 
  SCENARIO_SCENARIO="${SCENARIO_DIR}/$scenario_basename" 
  for town_scenario in "${SCENARIO_SCENARIO}"/*
  do 
    # scenario_town_basename="${town_scenario%.*}"
    scenario_town_basename=$(basename ${town_scenario%.*})
    # if [[ "$scenario_town_basename" = "Town01_Scenario1"  ||  "$scenario_town_basename" = "Town02_Scenario1" ]]; then
    #   continue
    # fi
    # echo "Town scenario basename: ${scenario_town_basename}"
    export SCENARIOS="${SCENARIO_SCENARIO}/${scenario_town_basename}.json"
    export ROUTES="${SCENARIO_ROUTES}/${scenario_town_basename}.xml"
    export CHECKPOINT_ENDPOINT="${WORK_DIR}/results/${scenario_town_basename}.json"
    export SAVE_PATH="${WORK_DIR}/results/${scenario_town_basename}"
    python3 ${LEADERBOARD_ROOT}/leaderboard/collect_cosim.py \
    --scenarios=${SCENARIOS}  \
    --routes=${ROUTES} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --resume=${RESUME}
  done
done

# python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
# python3 ${LEADERBOARD_ROOT}/leaderboard/collect_cosim.py \
# --scenarios=${SCENARIOS}  \
# --routes=${ROUTES} \
# --repetitions=${REPETITIONS} \
# --track=${CHALLENGE_TRACK_CODENAME} \
# --checkpoint=${CHECKPOINT_ENDPOINT} \
# --agent=${TEAM_AGENT} \
# --agent-config=${TEAM_CONFIG} \
# --debug=${DEBUG_CHALLENGE} \
# --resume=${RESUME}
