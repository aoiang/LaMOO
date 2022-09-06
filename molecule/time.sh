#!/bin/bash



python MCTS.py --data_id 0 --sample_num 10 --iter 85 --sample_method cmaes --cmaes_method vanilla > vcames_time0.txt
python MCTS.py --data_id 1 --sample_num 10 --iter 85 --sample_method cmaes --cmaes_method vanilla > vcames_time1.txt
python MCTS.py --data_id 2 --sample_num 10 --iter 85 --sample_method cmaes --cmaes_method vanilla > vcames_time2.txt
python MCTS.py --data_id 3 --sample_num 10 --iter 85 --sample_method cmaes --cmaes_method vanilla > vcames_time3.txt
python MCTS.py --data_id 4 --sample_num 10 --iter 85 --sample_method cmaes --cmaes_method vanilla > vcames_time4.txt



python MCTS.py --data_id 0 --sample_num 10 --iter 85 --sample_method cmaes > cames_time0.txt
python MCTS.py --data_id 1 --sample_num 10 --iter 85 --sample_method cmaes > cames_time1.txt
python MCTS.py --data_id 2 --sample_num 10 --iter 85 --sample_method cmaes > cames_time2.txt
python MCTS.py --data_id 3 --sample_num 10 --iter 85 --sample_method cmaes > cames_time3.txt
python MCTS.py --data_id 4 --sample_num 10 --iter 85 --sample_method cmaes > cames_time4.txt



#python MCTS.py --data_id 30 --sample_num 10 --iter 85 --cmaes_method lamcts --split_method dominance > do_bayesian_time0.txt
#python MCTS.py --data_id 31 --sample_num 10 --iter 85 --cmaes_method lamcts --split_method dominance > do_bayesian_time1.txt
#python MCTS.py --data_id 32 --sample_num 10 --iter 85 --cmaes_method lamcts --split_method dominance > do_bayesian_time2.txt
#python MCTS.py --data_id 33 --sample_num 10 --iter 85 --cmaes_method lamcts --split_method dominance > do_bayesian_time3.txt
#python MCTS.py --data_id 34 --sample_num 10 --iter 85 --cmaes_method lamcts --split_method dominance > do_bayesian_time4.txt



