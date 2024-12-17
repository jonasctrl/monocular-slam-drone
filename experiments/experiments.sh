#!/bin/bash

#goals=("40 50 3" "-70 80 2" "10 -100 3" "30 -49 3" "-30 -99 3" "-180 -90 2" "240 180 4" ) 
goals=("-30 -99 3"  "10 -100 3" "30 -49 3" "-180 -90 2") 
resolutions=("0.5" "1" "1.5" "2" "3" "4") 

#resolutions=("1.5") 

idx=0

#publish_occup="--publish_occup"
publish_occup=""

for goal in "${goals[@]}"; do
    echo "goal=${goal}"
	for resolution in "${resolutions[@]}"; do
		for iter in $(seq 3); do
			echo "resolution=$resolution"
			logfile="./results/log_${idx}.json"

			./mapper_nav_ros.py --reset_sim ${publish_occup} --logfile ${logfile} --goal_off ${goal} --map_resolution ${resolution} --exit_on_goal --unf_pos_tol_incr 1 --max_a_star_iters 1000 --unf_max_iters_incr 200 --max_steps 5000

			idx=$((idx+1))
		done
	done
done
