import argparse
from env import Env
from dqn import QNetwork
from tqdm import tqdm
import torch
import random
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
# Arguments used for ENV
parser.add_argument("--save_state", type=str, default="start_game.state", help="path to save_state file")
parser.add_argument("--rom", type=str, default="Pokemon Red.gb", help="path to rom file")
parser.add_argument("--epoch", type=int, default=100, help="number of epochs")
parser.add_argument("--lr", type=float, default=.01, help="learning rate")
parser.add_argument("--buffer", type=int, default=50, help="size of buffer")
parser.add_argument("--discount", type=float, default=.999, help="discount factor")
parser.add_argument("--training_steps", type=int, default=100, help="number of training steps")
parser.add_argument("--num_frames", type=int, default=1000000, help="max frames")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--train_limit_buffer", type=int, default=50, help="Only train after certain buffer?")
parser.add_argument("--conv_layers", type=int, default=16, help="number of conv layers")

args = parser.parse_args()


def get_action_dqn(network, state, epsilon, epsilon_decay):
    """Select action according to e-greedy policy and decay epsilon

      Args:
          network (QNetwork): Q-Network
          state (np-array): current state, size (state_size)
          epsilon (float): probability of choosing a random action
          epsilon_decay (float): amount by which to decay epsilon

      Returns:
          action (int): chosen action [0, action_size)
          epsilon (float): decayed epsilon
    """
    if random.random() < epsilon:
        action = random.randint(0, 1)
    else:
        with torch.no_grad():
            state = state.float().cuda()
            action = torch.argmax(network(state)).item()
    return action, epsilon * epsilon_decay


def prepare_batch(memory, batch_size):
    """Randomly sample batch from memory
       Prepare cuda tensors

      Args:
          memory (list): state, action, next_state, reward, done tuples
          batch_size (int): amount of memory to sample into a batch

      Returns:
          state (tensor): float cuda tensor of size (batch_size x state_size()
          action (tensor): long tensor of size (batch_size)
          next_state (tensor): float cuda tensor of size (batch_size x state_size)
          reward (tensor): float cuda tensor of size (batch_size)
          done (tensor): float cuda tensor of size (batch_size)
    """
    batch = random.sample(memory, batch_size)
    state = torch.tensor([x[0] for x in batch], dtype=torch.float).cuda()
    action = torch.tensor([x[1] for x in batch], dtype=torch.long).cuda()
    next_state = torch.tensor([x[2] for x in batch], dtype=torch.float).cuda()
    reward = torch.tensor([x[3] for x in batch], dtype=torch.float).cuda()
    done = torch.tensor([x[4] for x in batch], dtype=torch.float).cuda()
    return state, action, next_state, reward, done


def learn_dqn(batch, optim, q_network, target_network, gamma, global_step, target_update):
    """Update Q-Network according to DQN Loss function
       Update Target Network every target_update global steps

      Args:
          batch (tuple): tuple of state, action, next_state, reward, and done tensors
          optim (Adam): Q-Network optimizer
          q_network (QNetwork): Q-Network
          target_network (QNetwork): Target Q-Network
          gamma (float): discount factor
          global_step (int): total steps taken in environment
          target_update (int): frequency of target network update
    """
    state, action, next_state, reward, done = batch
    optim.zero_grad()
    values = q_network(state).gather(1, action.unsqueeze(-1)).squeeze()
    target = reward + gamma * (1 - done) * torch.max(target_network(next_state), 1)[0]
    loss = torch.mean((target - values) ** 2)
    loss.backward()
    optim.step()

    if global_step % target_update == 0:
        target_network.load_state_dict(q_network.state_dict())


def dqn_main():
    # Hyper parameters
    lr = 1e-3
    epochs = args.epoch
    start_training = 1000
    gamma = 0.99
    batch_size = args.batch_size
    epsilon = 1
    epsilon_decay = .9999
    target_update = 1000
    learn_frequency = 2
    env = Env(args)

    # Init environment
    state_size = env.state_size
    action_size = env.action_space


    # Init networks
    q_network = QNetwork(state_size, action_size).cuda()
    target_network = QNetwork(state_size, action_size).cuda()
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr=lr)

    # Init replay buffer
    memory = []

    # Begin main loop
    results_dqn = []
    global_step = 0
    for epoch in range(epochs):

        # Reset environment
        state = env.reset()
        done = False
        cum_reward = 0  # Track cumulative reward per episode
        frame = 0
        # Begin episode
        for step in tqdm(range(args.num_frames), desc='Episode: {} Reward: {}'.format(epoch, cum_reward)):
            if done or cum_reward <= -200:
                break
            frame += 1
            # Select e-greedy action
            action, epsilon = get_action_dqn(q_network, state, epsilon, epsilon_decay)

            # Take step
            next_state, reward, done, _ = env.step(action)
            if done:
                print('DONE!')
            # env.render()

            # Store step in replay buffer
            memory.append((state.numpy(), action, next_state.numpy(), reward, done))

            cum_reward += reward
            global_step += 1  # Increment total steps
            state = next_state  # Set current state

            # If time to train
            if global_step > start_training and global_step % learn_frequency == 0:
                # Sample batch
                batch = prepare_batch(memory, batch_size)

                # Train
                learn_dqn(batch, optim, q_network, target_network, gamma, global_step, target_update)

        # Print results at end of episode
        results_dqn.append(cum_reward)

    return results_dqn


results_dqn = dqn_main()

plt.plot(results_dqn)
plt.savefig('results.png')
