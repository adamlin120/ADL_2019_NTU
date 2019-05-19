def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''

    # general
    parser.add_argument('--seed', default=13, help='random seed', type=int)
    parser.add_argument('--model_path', default='', type=str)

    # DQN
    parser.add_argument('--dqn_batch_size', default=32, type=int)
    parser.add_argument('--dqn_net', default='DQN', type=str)
    parser.add_argument('--DoubleDQN', default=False, action='store_true')
    parser.add_argument('--dqn_lr', default=1e-4, type=float)
    parser.add_argument('--GAMMA', default=0.99, type=float)
    parser.add_argument('--train_freq', default=4, type=int)
    parser.add_argument('--target_update_freq', default=1000, type=int)
    parser.add_argument('--learning_start', default=10000, type=int)
    parser.add_argument('--num_timesteps', default=30000000, type=int)
    parser.add_argument('--display_freq', default=30, type=int)
    parser.add_argument('--save_freq', default=20000, type=int)
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--exploration_method', default='epsilon', type=str)
    parser.add_argument('--boltzmann_temperature', default=0.000001, type=float)
    parser.add_argument('--target_score', default=120, type=int)
    
    # PPO
    parser.add_argument('--PPO', action='store_true')
    parser.add_argument('--n_latent_var', default=64, type=int)
    parser.add_argument('--ppo_lr', default=0.0001, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--eps_clip', default=0.2, type=float)
    parser.add_argument('--K_epochs', default=5, type=int)
    parser.add_argument('--update_timestep', default=5000, type=int)

    return parser
