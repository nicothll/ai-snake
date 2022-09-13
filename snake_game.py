from typing import Tuple
import pygame
import random
from enum import Enum
from dataclasses import dataclass

pygame.init()
font = pygame.font.Font("./fonts/arial.ttf", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


@dataclass
class Point:
    x: int
    y: int


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)

# Parameters
BLOCK_SIZE: int = 20
SPEED: int = 40


class SnakeGame:
    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.w = width
        self.h = height

        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        # Init Game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 4)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score: int = 0
        self.food = None
        self._place_food()

    def _place_food(self) -> None:
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self) -> Tuple[bool, int]:  # TODO:
        # 1. Collect User Input
        # 2. Move
        # 3. Check if Game Over
        # 4. Place new food or just move
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. Return Game Over and Score
        game_over = False
        return game_over, self.score

    def _update_ui(self) -> None:
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display,
                BLUE1,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.display,
                BLUE2,
                pygame.Rect(pt.x + 4, pt.y + 4, 12, 12),
            )
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        # Update display
        pygame.display.flip()


def main():
    game = SnakeGame()

    while True:
        game_over, score = game.play_step()

        if game_over:
            break

    print(f"Final Score: {score}")

    pygame.quit()


if __name__ == "__main__":
    main()
