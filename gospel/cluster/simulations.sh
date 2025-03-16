#!/bin/bash

for thresold in 0.2 0.1; do
  for p_err in 0.1 0.2 0.3 0.4 0.5 0.6; do
    PORT=$((24395 + RANDOM % 1000))  # Generate a random port

    # Print the threshold, p, and assigned port
    echo "Running with thresold=$thresold, p_err=$p_err, PORT=$PORT"

    # Run the process in the background
    time python -m gospel.cluster.run_veriphix 100 100 100 $thresold $p_err \
      --walltime 2 --memory 4 --cores 4 --port $PORT --scale 100 &

  done
done

wait  # Ensure all background jobs complete

echo "All jobs completed!"
