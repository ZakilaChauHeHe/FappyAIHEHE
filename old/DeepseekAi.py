import pygame
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pygame.locals import *

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.fc(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Flappy Bird Game
class FlappyBird:
    def __init__(self):
        self.WIDTH = 400
        self.HEIGHT = 600
        self.BIRD_SIZE = 30
        self.PIPE_WIDTH = 50
        self.PIPE_GAP = 150
        self.GRAVITY = 0.5
        self.JUMP_VEL = -8
        self.PIPE_SPEED = 3
        
        self.reset()
    
    def reset(self):
        self.bird_y = self.HEIGHT // 2
        self.bird_vel = 0
        self.pipes = []
        self.score = 0
        self.add_pipe()
        self.add_pipe()
        return self.get_state()
    
    def add_pipe(self):
        gap_y = random.randint(100, self.HEIGHT - 100 - self.PIPE_GAP)
        self.pipes.append({
            'x': self.WIDTH,
            'top': gap_y - self.PIPE_GAP,
            'bottom': gap_y
        })
    
    def get_state(self):
        if len(self.pipes) > 0:
            next_pipe = self.pipes[0]
            dx = next_pipe['x'] - (self.WIDTH // 4)
            dy_top = self.bird_y - next_pipe['top']
            dy_bottom = self.bird_y - next_pipe['bottom']
        else:
            dx = 0
            dy_top = 0
            dy_bottom = 0
        
        return np.array([
            self.bird_y / self.HEIGHT,
            self.bird_vel / 10,
            dx / self.WIDTH,
            dy_top / self.HEIGHT,
            dy_bottom / self.HEIGHT
        ], dtype=np.float32)
    
    def step(self, action):
        if action == 1:
            self.bird_vel = self.JUMP_VEL
        
        self.bird_vel += self.GRAVITY
        self.bird_y += self.bird_vel
        
        reward = 0.1
        done = False
        
        if self.bird_y < 0 or self.bird_y > self.HEIGHT:
            done = True
            reward = -10
        
        for pipe in self.pipes:
            pipe['x'] -= self.PIPE_SPEED
            if pipe['x'] + self.PIPE_WIDTH < 0:
                self.pipes.remove(pipe)
                if pipe == self.pipes[0]:
                    self.score += 1
                    reward = 5
                    self.add_pipe()
            
            if (pipe['x'] < self.WIDTH//4 + self.BIRD_SIZE < pipe['x'] + self.PIPE_WIDTH):
                if not (pipe['top'] < self.bird_y < pipe['bottom']):
                    done = True
                    reward = -10
        
        return self.get_state(), reward, done

# DQN Agent
class DQNAgent:
    def __init__(self, input_size, output_size):
        self.policy_net = DQN(input_size, output_size)
        self.target_net = DQN(input_size, output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return torch.argmax(q_values).item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Training Parameters
NUM_AGENTS = 25
GENERATIONS = 1000
UPDATE_TARGET_EVERY = 10

pygame.init()
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# Create single window for both game and stats
window = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Flappy Bird AI Training")

agent = DQNAgent(5, 2)
envs = [FlappyBird() for _ in range(NUM_AGENTS)]
scores = []
avg_scores = []

generation = 0
sim_speed = 5
training = True

def draw_game(env, x_offset):
    # Draw bird
    bird_rect = pygame.Rect(
        x_offset + env.WIDTH//4 - env.BIRD_SIZE//2,
        env.bird_y - env.BIRD_SIZE//2,
        env.BIRD_SIZE,
        env.BIRD_SIZE
    )
    pygame.draw.rect(window, (255, 255, 0), bird_rect)
    
    # Draw pipes
    for pipe in env.pipes:
        # Top pipe
        pygame.draw.rect(window, (0, 255, 0), (
            x_offset + pipe['x'],
            0,
            env.PIPE_WIDTH,
            pipe['top']
        ))
        # Bottom pipe
        pygame.draw.rect(window, (0, 255, 0), (
            x_offset + pipe['x'],
            pipe['bottom'],
            env.PIPE_WIDTH,
            env.HEIGHT - pipe['bottom']
        ))

while generation < GENERATIONS and training:
    states = [env.reset() for env in envs]
    dones = [False] * NUM_AGENTS
    total_rewards = [0] * NUM_AGENTS
    
    while not all(dones):
        window.fill((0, 0, 0))
        
        for event in pygame.event.get():
            if event.type == QUIT:
                training = False
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    sim_speed = min(10, sim_speed + 1)
                if event.key == K_DOWN:
                    sim_speed = max(1, sim_speed - 1)
        
        actions = [agent.act(state) for state in states]
        
        next_states = []
        rewards = []
        new_dones = []
        for i in range(NUM_AGENTS):
            if not dones[i]:
                ns, r, d = envs[i].step(actions[i])
                next_states.append(ns)
                rewards.append(r)
                new_dones.append(d)
                total_rewards[i] += r
            else:
                next_states.append(states[i])
                rewards.append(0)
                new_dones.append(True)
        
        for i in range(NUM_AGENTS):
            if not dones[i]:
                agent.buffer.push(states[i], actions[i], rewards[i], next_states[i], new_dones[i])
        
        states = next_states
        dones = new_dones
        
        agent.update_model()
        
        # Draw all agents
        window.fill((0, 0, 0))
        for i, env in enumerate(envs[:4]):  # Show first 4 agents
            draw_game(env, (i % 2) * 400)
        
        # Draw stats
        stats_text = [
            f"Generation: {generation+1}/{GENERATIONS}",
            f"Speed: {sim_speed}x",
            f"Epsilon: {agent.epsilon:.3f}",
            f"Avg Score: {np.mean([e.score for e in envs]):.1f}",
            f"Max Score: {max([e.score for e in envs])}"
        ]
        
        y = 10
        for text in stats_text:
            surf = font.render(text, True, (255, 255, 255))
            window.blit(surf, (500, y))
            y += 40
        
        pygame.display.flip()
        clock.tick(30 * sim_speed)
    
    if generation % UPDATE_TARGET_EVERY == 0:
        agent.update_target_net()
    
    agent.update_epsilon()
    
    # Print progress
    game_scores = [env.score for env in envs]
    avg_game_score = sum(game_scores) / NUM_AGENTS
    max_game_score = max(game_scores)
    print(f"Iteration {generation+1}/{GENERATIONS} - Avg Score: {avg_game_score:.2f} | Max Score: {max_game_score} | Epsilon: {agent.epsilon:.3f}")
    
    generation += 1
    avg_scores.append(avg_game_score)

pygame.quit()