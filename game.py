import pygame
import sys
import random

import Config

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
PIPE_WIDTH = 50
GAP_SIZE = 150
PIPE_SPACING = 300
PIPE_SPEED = 3
GRAVITY = 0.5
FLAP_STRENGTH = 10
FPS = 60
SKY_BLUE = (135, 206, 235)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

# Clock to control frame rate
clock = pygame.time.Clock()

# Font for score and game over text
font = pygame.font.Font(None, 36)


# Bird class
class Bird:
    def __init__(self):
        self.x = 100
        self.y = SCREEN_HEIGHT / 2
        self.velocity = 0
        self.rect = pygame.Rect(self.x, self.y, 30, 30)  # 30x30 bird

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.y = self.y
        if self.y < 0:
            self.y = 0
            self.velocity = 0

    def flap(self):
        self.velocity = -FLAP_STRENGTH

    def draw(self, screen):
        pygame.draw.rect(screen, YELLOW, self.rect)


# Pipe class
class Pipe:
    def __init__(self, x, gap_y):
        self.x = x
        self.gap_y = gap_y
        self.top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.gap_y - GAP_SIZE / 2)
        self.bottom_rect = pygame.Rect(
            self.x,
            self.gap_y + GAP_SIZE / 2,
            PIPE_WIDTH,
            SCREEN_HEIGHT - (self.gap_y + GAP_SIZE / 2),
        )
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.top_rect)
        pygame.draw.rect(screen, GREEN, self.bottom_rect)


class Game:
    def __init__(self) -> None:
        # Set up the game window
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")

        self.bird = Bird()
        self.pipes: list[Pipe] = []
        self.score = 0
        self.game_over = False

    def _getstate(self) -> object:
        try:
            next_pipe = self.pipes[0]
        except IndexError:
            next_pipe = Pipe(SCREEN_WIDTH, SCREEN_HEIGHT / 2)
        state = [
            self.bird.y / SCREEN_HEIGHT,
            (next_pipe.gap_y - self.bird.y) / SCREEN_HEIGHT,
            (next_pipe.x - self.bird.x) / SCREEN_WIDTH,
        ]

    # Function to reset the game
    def reset_game(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.game_over = False
        return self._getstate()

    def step(self, action):
        if action:
            self.bird.flap()

        reward = Config.SURVIVE_TIME_REWARD
        # Handle events

        if not self.game_over:
            # Update bird
            self.bird.update()

            # Update pipes
            for pipe in self.pipes:
                pipe.update()

            # Add new pipe if needed
            if len(self.pipes) == 0 or self.pipes[-1].x < SCREEN_WIDTH - PIPE_SPACING:
                gap_y = random.randint(100, SCREEN_HEIGHT - 100)
                self.pipes.append(Pipe(SCREEN_WIDTH, gap_y))

            # Remove pipes that are off-screen
            self.pipes = [pipe for pipe in self.pipes if pipe.x > -PIPE_WIDTH]

            # Check for collisions
            if self.bird.rect.bottom >= SCREEN_HEIGHT:
                self.game_over = True
                reward += Config.HIT_BOTTOM_REWARD
            for pipe in self.pipes:
                if self.bird.rect.colliderect(
                    pipe.top_rect
                ) or self.bird.rect.colliderect(pipe.bottom_rect):
                    self.game_over = True
                    reward += Config.HIT_PIPE_REWARD

            # Update score
            for pipe in self.pipes:
                if not pipe.passed and self.bird.x > pipe.x + PIPE_WIDTH / 2:
                    self.score += 1
                    reward += Config.PASS_PIPE_REWARD
                    pipe.passed = True

        # Update the display
        clock.tick(FPS)

        return self._getstate(), reward, self.game_over

    def render(self):
        # Draw everything
        self.screen.fill(SKY_BLUE)  # Background
        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.bird.draw(self.screen)

        # Draw score
        score_text = font.render(str(self.score), True, WHITE)
        self.screen.blit(
            score_text, (SCREEN_WIDTH / 2 - score_text.get_width() / 2, 50)
        )

        # Draw game over text if game is over
        if self.game_over:
            game_over_text = font.render(
                "Game Over! Press space to restart", True, WHITE
            )
            self.screen.blit(
                game_over_text,
                (
                    SCREEN_WIDTH / 2 - game_over_text.get_width() / 2,
                    SCREEN_HEIGHT / 2,
                ),
            )
        pygame.display.flip()
