from tqdm import tqdm
from torchvision import transforms
import torch
from pyboy import PyBoy, windowevent
import io


class Env:
    def __init__(self, args):
        self.locations = []
        self.events = ['D5F3', 'D60D', 'D700', 'D70B', 'D70C', 'D70D', 'D70E', 'D710', 'D714', 'D72E',
                       'D751', 'D755', 'D75E', 'D773', 'D77C', 'D782', 'D790', 'D792', 'D79A', 'D7B3', 'D7D4',
                       'D7D8', 'D7E0', 'D7EE', 'D803', 'D85F']
        self.towns_visited = 0
        self.event_tot = 0
        self.badges = 0
        self.owned_pokemon = 0
        self.position = {}
        self.actions_since_moved = 0
        self.pyboy = PyBoy(
            args.rom,
            # window_type="headless",  # For unattended use, for example machine learning
        )
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.args = args
        self.action_space = 8
        self.pyboy.set_emulation_speed(0)
        self.buffered_state = io.BytesIO()
        if args.save_state != "":
            self.pyboy.load_state(open(args.save_state, 'rb'))
            self.pyboy.save_state(self.buffered_state)
        else:
            for n in tqdm(range(4000)):  # Move ahead the desired number of frames.
                self.pyboy.tick()
                if 1298 == n:
                    self.pyboy.send_input(windowevent.PRESS_BUTTON_START)
                elif n == 1320:
                    self.pyboy.send_input(windowevent.RELEASE_BUTTON_START)
                elif n > 1320 and n % 2 == 0:
                    self.pyboy.send_input(windowevent.PRESS_BUTTON_A)
                elif n > 1320 and n % 2 != 0:
                    self.pyboy.send_input(windowevent.RELEASE_BUTTON_A)
            self.pyboy.save_state(open('start_game.state', 'wb'))
            self.pyboy.save_state(self.buffered_state)

        print(self.pyboy.get_memory_value(0xcc24))
        self.position['x'] = self.pyboy.get_memory_value(0xd361)
        self.position['y'] = self.pyboy.get_memory_value(0xd362)

    def step(self, action):
        frames = []
        for i in range(8):
            self.pyboy.send_input(action+1) #Translate up 1 action because 0 is quit
            self.pyboy.tick()
            frames.append(self.transform(self.pyboy.get_screen_image()))
        self.pyboy.send_input(action + 9) #Translate up 9 actions to release
        self.pyboy.tick()
        frames.append(self.transform(self.pyboy.get_screen_image()))
        frames = torch.stack(frames, 0)
        if self.actions_since_moved == 25:
            self.reset()
            return frames, 0, True, False
        return frames, self.getReward(action + 1), False, False

    def reset(self):
        self.buffered_state.seek(0)
        self.pyboy.load_state(self.buffered_state)
        self.pyboy.tick()
        self.towns_visited = 0
        self.event_tot = 0
        self.badges = 0
        self.owned_pokemon = 0
        self.position = {}
        self.actions_since_moved = 0
        self.position['x'] = self.pyboy.get_memory_value(0xd361)
        self.position['y'] = self.pyboy.get_memory_value(0xd362)
        return self.transform(self.pyboy.get_screen_image()).unsqueeze(0)

    def getReward(self, action):
        # Event Flags +1 D5A6-D85F
        event_flags = 0
        reward = 0
        for event in self.events:
            event_flags += self.pyboy.get_memory_value(int(event, 16))
        if event_flags > self.event_tot:
            self.event_tot = event_flags
            reward += 1
        # New Location +1 TownVisitedFlag d70b 13 flags (max 13 towns)
        towns_visited = self.pyboy.get_memory_value(int('d70b', 16))
        if towns_visited > self.towns_visited:
            self.towns_visited = towns_visited
            reward += 1
        # Faint Enemy +1
        # Faint player pokemon -1
        if self.inBattle():
            self.actions_since_moved = 0
            # Win Battle +1 CF0B 00=win
            # Lost Battle -1 CF0B 01=lose 02=draw
            outcome = self.pyboy.get_memory_value(int('cf0b', 16))
            if outcome == 0:
                reward += 1
            else:
                reward -= 1
        # Badge +1 D356
        badges = self.pyboy.get_memory_value(int('d356', 16))
        if badges > self.badges:
            self.badges += 1
            reward += 1
        # Own new pokemon +1? d2f7-d31c
        owned_pokemon = 0
        for address in range(0xd2f7, 0xd31d):
            owned_pokemon += self.pyboy.get_memory_value(address)
        if owned_pokemon > self.owned_pokemon:
            self.owned_pokemon += owned_pokemon
            reward += 1
        # Blackout d12d nonzero if blacked out outside of battle
        if self.pyboy.get_memory_value(0xd12d) != 0:
            reward -= 1
        if not self.inBattle():
            if self.position['x'] == self.pyboy.get_memory_value(0xd361) and self.position['y'] == self.pyboy.get_memory_value(0xd362):
                self.actions_since_moved += 1
            else:
                self.actions_since_moved = 0
                self.position['x'] = self.pyboy.get_memory_value(0xd361)
                self.position['y'] = self.pyboy.get_memory_value(0xd362)
            if action in range(0, 4):
                reward += .01
            reward -= self.actions_since_moved * .005
        return reward

    def inBattle(self):
        # IsInBattle d057 0 = no battle
        return self.pyboy.get_memory_value(0xd057) != 0



