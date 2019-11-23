import argparse
from env import Env
from dqn import DQN
from tqdm import tqdm
parser = argparse.ArgumentParser()
# Arguments used for ENV
parser.add_argument("--save_state", type=str, default="", help="path to save_state file")
parser.add_argument("--rom", type=str, default="Pokemon Red.gb", help="path to rom file")
parser.add_argument("--epoch", type=int, default=100, help="number of epochs")
parser.add_argument("--lr", type=float, default=.01, help="learning rate")
parser.add_argument("--buffer", type=int, default=50, help="size of buffer")
parser.add_argument("--discount", type=float, default=.999, help="discount factor")
parser.add_argument("--training_steps", type=int, default=100, help="number of training steps")
parser.add_argument("--num_frames", type=int, default=100000, help="max frames")
parser.add_argument("--batch_size", type=int, default=50, help="batch size")
parser.add_argument("--train_limit_buffer", type=int, default=50, help="Only train after certain buffer?")
parser.add_argument("--conv_layers", type=int, default=16, help="number of conv layers")

args = parser.parse_args()
env = Env(args)
state = env.reset()
model = DQN(env, num_epochs=args.epoch, learning_rate=args.lr, buffer_size=args.buffer, discount_factor=args.discount,
            num_training_steps=args.training_steps,
            action_space=env.action_space, num_frames=args.num_frames, batch_size=args.batch_size, num_conv_layers=16,
            input_channels=state.size(-3), pool_kernel_size=3,
            kernel_size=3, dense_layer_features=256,
            height=state.size(-2), width=state.size(-1), train_limit_buffer=args.train_limit_buffer, use_cuda=True)
for _ in tqdm(range(args.epoch)):
    model.train()
# For unattended use, the screen buffer can be displayed using the following:
