import math
import random
from collections import namedtuple

import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

import settings
from ai.ai_player_rl import AI_PlayerRL
from asteroids.app import App
from asteroids.asteroid import Asteroid
from asteroids.utils import render_on, WHITE

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(8, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        return self.linear4(x)


class AI_AppRL(App):
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    def __init__(self, state_dict):
        super(AI_AppRL, self).__init__()
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

        device = torch.device('cuda')

        policy_net = DQN().to(device)
        if state_dict is not None:
            policy_net.load_state_dict(state_dict)
        target_net = DQN().to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.RMSprop(policy_net.parameters())
        memory = ReplayMemory(10000)

        self.device = device

        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.optimizer = optimizer

        self.steps_done = 0
        self.episode_durations = []

    def _spawn_player(self):
        """
        Creates and returns a new AI_Player in the center of the screen.
        """
        return AI_PlayerRL(settings.WIDTH / 2, settings.HEIGHT / 2)

    def _update_player(self, action):
        """
        Reads the current game state + has the player respond accordingly.
        """
        player = self.player  # type: AI_PlayerRL
        decision_vector = [0, 0, 0, 0]
        decision_vector[action] = 1
        decision_vector = list(map(bool, decision_vector))
        player.update(self.bullets, None)
        player.perform_decisions(decision_vector, self.bullets)

    def _render_ai_spectator_overlay(self):
        """
        Renders overlay components used in AI Spectator mode.
        Returns a list of rectangles to be re-rendered.
        """
        render_rects = []

        # Show fitness stats in top-left, under Score, if necessary
        if settings.SHOW_SCORE:
            run_time_text = self._small_font.render("Runtime: %ds (%d)" %
                                                    (self.run_time / 60, self.run_time),
                                                    True,
                                                    WHITE)
            run_time_rect = render_on(run_time_text,
                                      self.screen,
                                      run_time_text.get_width() / 2,
                                      run_time_text.get_height() * 3 / 2)
            render_rects.append(run_time_rect)

            accuracy_text = self._small_font.render("Accuracy: %.2f" %
                                                    self._get_accuracy(),
                                                    True,
                                                    WHITE)
            accuracy_rect = render_on(accuracy_text,
                                      self.screen,
                                      accuracy_text.get_width() / 2,
                                      accuracy_text.get_height() * 5 / 2)
            render_rects.append(accuracy_rect)

            fitness_text = self._small_font.render("Fitness: %d" %
                                                   self._get_fitness(),
                                                   True,
                                                   WHITE)
            fitness_rect = render_on(fitness_text,
                                     self.screen,
                                     fitness_text.get_width() / 2,
                                     fitness_text.get_height() * 7 / 2)
            render_rects.append(fitness_rect)

            generation_text = self._small_font.render("Generation: %d" %
                                                      self.generation,
                                                      True,
                                                      WHITE)
            generation_rect = render_on(generation_text,
                                        self.screen,
                                        generation_text.get_width() / 2,
                                        generation_text.get_height() * 9 / 2)
            render_rects.append(generation_rect)

        # Return the rects to be re-rendered
        return render_rects

    def _handle_ai_spectator_controls(self, event):
        """
        Checks whether event was an AI Spectator mode
        specific control, and handles it if so.
        """
        pass

    def _get_accuracy(self):
        """
        Returns the player's current accuracy.
        """
        if self.player.num_bullets_fired == 0:
            return 0.0
        return 1.0 * self.asteroids_hit / self.player.num_bullets_fired

    def _get_fitness(self):
        """
        Returns the current fitness score.
        """
        return ((self.score * settings.FITNESS_SCORE_WEIGHT) +
                (self.run_time * settings.FITNESS_RUN_TIME_WEIGHT) +
                (self._get_accuracy() * settings.FITNESS_ACCURACY_WEIGHT))

    def start_game(self, ai_brain):
        """
        Starts the game using the provided AI controller.
        """
        self._ai_brain = ai_brain
        super(AI_AppRL, self).start_game()

    def _update(self):
        """
        Performs one step of the execution loop for all game components.
        """
        if self._state != App.RUNNING and self._state != App.GAME_OVER:
            return

        # If the player is destroyed, transition to Game Over state or quit.
        if self._state == App.RUNNING and self.player.destroyed:
            # if self._use_screen:
            #     self._load_game_over()
            # else:
            self._running = False

            # Restore RNG state as well if necessary
            if settings.USE_PREDETERMINED_SEED:
                random.setstate(self._prev_rng_state)

        # Remove destroyed components
        self.bullets = list(filter(lambda x: not x.destroyed, self.bullets))
        self.asteroids = list(filter(lambda x: not x.destroyed, self.asteroids))

        # Get the approximate number of milliseconds since last asteroid spawn
        # Since the game operates on frames (the number of update iterations),
        # we multiply the frame difference by (1000 ms / s) * (1s / 60 frames)
        frames_since_last_spawn = (self.run_time - self._last_spawn_time)
        ms_since_last_spawn = frames_since_last_spawn * 1000.0 / 60.0

        # If the spawn period has expired, spawn a new aimed Asteroid
        if ms_since_last_spawn > self._spawn_period:
            new_spawn_period = self._spawn_period - settings.SPAWN_PERIOD_DEC
            self._spawn_period = max(new_spawn_period, settings.MIN_SPAWN_PERIOD)
            self._last_spawn_time = self.run_time
            Asteroid.spawn(self.asteroids, self.player, True)

        # Update the player with the current game state

        self.state = torch.tensor([self.last_sensor], device=self.device).float()

        action = self.select_action(self.state)
        # print(action)
        if not self.player.destroyed:
            self._update_player(action.item())  # put action here

        # Move all game components
        self.player.move()

        for bullet in self.bullets:
            bullet.move()
        for asteroid in self.asteroids:
            asteroid.move()

        # Check for player collisions with asteroids:
        self.player.check_for_collisions(self.asteroids)

        # Age and check for bullet collisions with asteroids
        curr_score = 0
        # acc = 0
        for bullet in self.bullets:
            bullet_score = bullet.check_for_collisions(self.asteroids)
            # distance = ((bullet.y - self.player.y) ** 2 + (bullet.x - self.player.x) ** 2) ** 0.5
            self.score += bullet_score
            self.asteroids_hit += int(bullet_score > 0)
            curr_score += int(bullet_score > 0)
            bullet.increase_age()

        # reward = float(curr_score * max(1. - self.player.speed / self.player.MAX_SPEED, 0.6))
        if self.player.destroyed:
            reward = float(-5)
        else:
            reward = float((0.5 - max(self.last_sensor)) * 10.0 + curr_score)

        # print(self.last_sensor.tolist())

        # self.last_fitness = self._get_fitness()

        reward = torch.tensor([reward], device=self.device)
        # Increment run time if the player is still alive
        if not self.player.destroyed:
            self.run_time += 1

        self.last_sensor = self.curr_sensor
        self.curr_sensor = self.player.sense(self.asteroids, self.bullets)

        if not self.player.destroyed:
            next_state = torch.tensor([self.curr_sensor], device=self.device).float()
            # print(next_state)
        else:
            next_state = None

        self.memory.push(self.state, action, next_state, reward)

        self.state = next_state

        self.optimize_model()

        self.t += 1
        if self.player.destroyed:
            self.episode_durations.append(curr_score)

    def run_simulation(self, generation):
        """
        Runs the game to completion in non-graphical mode using
        the provided AI controller, and returns the fitness score.
        """
        # Turn off sounds for the duration of the simulation
        previous_play_sfx = settings.PLAY_SFX
        settings.PLAY_SFX = False
        self.generation = generation

        # Prepare the simulation
        if not self._has_started:
            self._setup(use_screen=True)
        else:
            self._running = True

        self.t = 0
        self._load_level()

        self.last_fitness = 0
        self.last_sensor = self.player.sense(self.asteroids, self.bullets)
        self.curr_sensor = self.player.sense(self.asteroids, self.bullets)
        # Run it until the player dies
        while self._running:
            for event in pygame.event.get():
                self._handle_event(event)
            self._update()
            self._render()

        # Clean up the app for potential reuse
        settings.PLAY_SFX = previous_play_sfx

        # Return the fitness score
        return self._get_fitness()

    def cleanup_simulation(self):
        """
        Cleans up the app after all simulations are run.
        """
        self._cleanup()

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        if self.t == 0:
            print('loss', loss.item())
            print('scores', self.episode_durations[-1])
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # return torch.sigmoid(self.policy_net(state))
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long)
