# Monocular SLAM Drone

Update .env file with your system specific variables.

To build the project, run the following commands:

```bash
docker compose up --build
```

To stop the project, run the following command:

```bash
docker compose down
```

Mapper usage information can be accesed using:

```bash
./mapper_nav_ros.py --help
```

Example usage:

```bash
./mapper_nav_ros.py --reset_sim --publish_occup --logfile ./log.json --goal_off -111 100 3 --exit_on_goal  --unf_pos_tol_incr 1 --max_a_star_iters 500 --unf_max_iters_incr 200 --use_rgb_imaging --speed 2 --dimg_max_depth 20
```
