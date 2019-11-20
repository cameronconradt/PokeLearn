import argparse
from env import Env
from dqn import DQN

parser = argparse.ArgumentParser()
# Arguments used for ENV
parser.add_argument("--save_state", type=str, default="", help="path to save_state file")
parser.add_argument("--rom", type=str, default="Pokemon Red.gb", help="path to rom file")
parser.add_argument("--epoch", type=int, default=100, help="number of epochs")
parser.add_argument("--lr", type=float, default=.01, help="learning rate")
parser.add_argument("--buffer", type=int, default=50, help="size of buffer")

args = parser.parse_args()
env = Env(args)
state = env.reset()
model = DQN(env, num_epochs=args.epoch, learning_rate=args.lr, buffer_size=args.buffer, num_conv_layers=16,
            input_channels=state.size(-3),
            action_space=env.action_space, pool_kernel_size=3,
            kernel_size=3, dense_layer_features=256,
            height=state.size(-2), width=state.size(-1))
action_space = env.action_space

# For unattended use, the screen buffer can be displayed using the following:
