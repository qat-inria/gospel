#!/bin/bash

# rm -rf circuits
# python -m gospel.sampling_circuits.experiments

n_instances=100
bqp_error=0.4
# python -m gospel.cluster.generate_circuit_sample $n_instances $bqp_error


n_comp_run=100
n_test_run=100
n_nodes=$n_instances


# echo "GENTLE GLOBAL NOISE (remaining)"
# # Gentle global noise
# for p_err in 0.2 0.3 0.5 0.7 ; do
#   PORT=24395

#   # Print p and assigned port
#   echo "Running with p_err=$p_err, PORT=$PORT"

#   # Run the process in the background
#   time python -m gospel.cluster.run_veriphix $n_comp_run $n_test_run $n_instances $p_err $bqp_error --walltime 3 --memory 4 --cores 4 --port $PORT --scale $n_nodes

# done

echo "STRONG GLOBAL NOISE"
# Strong global noise
for p_err in 0.5 ; do
  PORT=24395

  # Print p and assigned port
  echo "Running with p_err=$p_err, PORT=$PORT"

  # Run the process in the background
  time python -m gospel.cluster.run_veriphix-strong $n_comp_run $n_test_run $n_instances $p_err $bqp_error --scale 16 #--walltime 3 --memory 4 --cores 4 --port $PORT --scale $n_nodes

done

# echo "DEPOLARIZING (CORRELATED)"
# # Depolarizing
# for p_err in 0.008 0.01 0.014 0.02 ; do
#   PORT=24395

#   # Print p and assigned port
#   echo "Running with p_err=$p_err, PORT=$PORT"

#   # Run the process in the background
#   time python -m gospel.cluster.run_veriphix-depol $n_comp_run $n_test_run $n_instances $p_err $bqp_error --walltime 3 --memory 4 --cores 4 --port $PORT --scale $n_nodes

# done

# echo "DEPOLARIZING (UNCORRELATED)"
# # Depolarizing
# for p_err in 0.03 ; do
#   PORT=24395

#   # Print p and assigned port
#   echo "Running with p_err=$p_err, PORT=$PORT"

#   # Run the process in the background
#   time python -m gospel.cluster.run_veriphix-uncorr_depol $n_comp_run $n_test_run $n_instances $p_err $bqp_error --walltime 3 --memory 4 --cores 4 --port $PORT --scale $n_nodes

# done

wait  # Ensure all background jobs complete

echo "All jobs completed!"
