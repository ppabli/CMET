#!/bin/bash

if [ -z "$1" ]; then
	echo "Usage: $0 <script_name> [num_iterations] [--load-env] [--no-env-deactivate]"
	exit 1
fi

script_name="$1"
num_ite=${2:-10}
load_env=true
deactivate_env=false

for arg in "$@"; do
	case $arg in
		--load-env)
			load_env=true
			;;
		--no-env-deactivate)
			deactivate_env=false
			;;
	esac
done

if [ "$load_env" = true ]; then
	echo "Activating virtual environment..."
	source "$HOME/Desktop/test/venv/bin/activate"
fi

echo "Running script: $script_name"
echo "Number of iterations: $num_ite"

for i in $(seq 1 $num_ite); do
	echo "Iteration $i"
	python "$script_name"
	sleep 1
done

echo "All iterations completed."

if [ "$load_env" = true ] && [ "$deactivate_env" = true ]; then
	echo "Deactivating virtual environment..."
	deactivate
fi
