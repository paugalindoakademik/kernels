#!/bin/bash
# @ job_name = MyJob
# @ initialdir = .
# @ output = %j.out
# @ error = %j.err
# @ wall_clock_limit = 00:10:00
# @ total_tasks = 1
# @ cpus_per_task = 6
# @ gpus_per_node = 1
# @ reservation = CUDA23-day1


./parboil run deviceQuery cuda default

