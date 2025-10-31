for ((i=0;i<5;i++)); do
	CUDA_VISIBLE_DEVICES=$i bash run.sh &
done
