import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 640, 480
PIPE_WIDTH = 80
PIPE_GAP = 150
BIRD_SIZE = 30

# Colors
BIRDCOLOR = (255,255,0)
PIPECOLOR = (124,252,0)
BGCOLOR = (135,206,250)

# Game variables
bird_y = HEIGHT // 2
pipe_x = WIDTH
pipe_y = random.randint(50, HEIGHT - PIPE_GAP - 50)

# Q-learning variables
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Q-table
q_table = {}

def get_state():
    """Simplify state to bird's y-position and pipe's x-position."""
    return (bird_y // 10, pipe_x // 10)

def choose_action(state, epsilon_greedy=True):
    """Choose action based on epsilon-greedy strategy."""
    if epsilon_greedy and random.random() < epsilon:
        return random.choice(['flap', 'do_nothing'])
    else:
        q_values = [q_table.get((state, 'flap'), 0), q_table.get((state, 'do_nothing'), 0)]
        return 'flap' if q_values[0] > q_values[1] else 'do_nothing'

def update_q_table(state, action, reward, next_state):
    """Update Q-table using Q-learning update rule."""
    q_value = q_table.get((state, action), 0)
    next_q_values = [q_table.get((next_state, 'flap'), 0), q_table.get((next_state, 'do_nothing'), 0)]
    next_q_value = max(next_q_values)
    new_q_value = q_value + alpha * (reward + gamma * next_q_value - q_value)
    q_table[(state, action)] = new_q_value

def draw_game(screen):
    """Draw game elements on the screen."""
    screen.fill(BGCOLOR)
    pygame.draw.rect(screen, PIPECOLOR, (pipe_x, 0, PIPE_WIDTH, pipe_y))
    pygame.draw.rect(screen, PIPECOLOR, (pipe_x, pipe_y + PIPE_GAP, PIPE_WIDTH, HEIGHT))
    pygame.draw.rect(screen, BIRDCOLOR, (WIDTH // 2, bird_y, BIRD_SIZE, BIRD_SIZE))
    pygame.display.update()

def reset_game():
    global running
    bird_y = HEIGHT // 2
    pipe_x = WIDTH
    pipe_y = random.randint(50, HEIGHT - PIPE_GAP - 50)
    
episodes = 100

def main():
    global bird_y, pipe_x, pipe_y
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    for episode in range(episodes):
        reset_game()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get current state
            state = get_state()

            # Choose action
            action = choose_action(state)

            # Apply action
            if action == 'flap':
                bird_y -= 20
            else:
                bird_y += 2  # Gravity

            # Update pipe position
            pipe_x -= 2
            if pipe_x < -PIPE_WIDTH:
                pipe_x = WIDTH
                pipe_y = random.randint(50, HEIGHT - PIPE_GAP - 50)

            # Check collision and reward
            reward = -1  # Default reward for living
            if (pipe_x < WIDTH // 2 + BIRD_SIZE and
                pipe_x + PIPE_WIDTH > WIDTH // 2 and
                (bird_y < pipe_y or bird_y > pipe_y + PIPE_GAP)):
                reward = -100  # Collision penalty
                running = False
            elif pipe_x < WIDTH // 2 and pipe_x + 2 >= WIDTH // 2:
                reward = 10  # Passing pipe reward

            # Get next state
            next_state = get_state()

            # Update Q-table
            update_q_table(state, action, reward, next_state)

            # Draw game
            draw_game(screen)

            # Cap framerate
            clock.tick(60)

    pygame.quit()



if __name__ == "__main__":
    main()