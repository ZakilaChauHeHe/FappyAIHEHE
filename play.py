import pygame
import sys
import game as g


def run():
    game = g.Game()

    gameover = False
    game.reset_game()

    while True:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1
                    if gameover:
                        game.reset_game()
        state, reward, gameover = game.step(action)
        game.render()
