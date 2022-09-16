from train_ensemble_agents import run_gym_ensemble
import argparse
import random
import torch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--CnnDQN", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--target_update_rate", type=float, default=0.1)
    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--start_train_ts", type=int, default=10000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=int, default=30000)
    parser.add_argument("--max_ts", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--log_every", type=int, default=10000)
    parser.add_argument("--target_network_update_f", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default='weights')
    parser.add_argument("--load_dir", type=str, default='weights')
    parser.add_argument("--save_transition", type=bool, default=False)
    parser.add_argument("--transition_steps", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=50000)
    parser.add_argument("--transition_dir", type=str, default='transitions')
    parser.add_argument("--ensemble_n", type=int, default=4)
    parser.add_argument("--H", type=int, default=20)
    parser.add_argument("--num_policies", type=int, default=10)
    parser.add_argument("--avf_buffer_size", type=int, default=10000)
    parser.add_argument("--ensembles_to_test", type=int, default=40)
    parser.add_argument("--avf_epochs", type=int, default=30)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_gym_ensemble(args)
        
