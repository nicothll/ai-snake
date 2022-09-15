from typing import Tuple
import pygame
import random
from enum import Enum
from dataclasses import dataclass

pygame.init()

SCORE_FONT = pygame.font.Font("./fonts/arial.ttf", 25)
GAME_OVER_FONT = pygame.font.Font("./fonts/arial.ttf", 60)


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
GRAY = (50, 50, 50)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)

# Parameters
BLOCK_SIZE: int = 20
SPEED: int = 10


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
        x = random.randint(0, (self.w - BLOCK_SIZE * 2) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE * 2) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self) -> Tuple[bool, int]:
        # 1. Collect User Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        # 2. Move
        self._move(self.direction)  # Update the head
        self.snake.insert(0, self.head)

        # 3. Check if Game Over
        game_over = False
        if self._collision():
            game_over = True
            return game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Return Game Over and Score
        return game_over, self.score

    def _collision(self):
        # Hits boundaries
        if (
            self.head.x > self.w - BLOCK_SIZE
            or self.head.x < 0
            or self.head.y > self.h - BLOCK_SIZE
            or self.head.y < 0
        ):
            return True
        # Hits itself
        if self.head in self.snake[1:]:
            return True

        return False

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

        text = SCORE_FONT.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])

        # Update display
        pygame.display.flip()

    def _move(self, direction):
        x, y = self.head.x, self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def show_game_over(self):
        # self.display.blit(GRAY, (0, 0))
        text_surface = GAME_OVER_FONT.render(f"GAME OVER", True, WHITE)
        text_rect = text_surface.get_rect(center=(self.w // 2, self.h // 2))
        self.display.blit(text_surface, text_rect)
        pygame.display.flip()

        # For restart (unset)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        return True


def main():

    game = SnakeGame()
    while True:
        game_over, score = game.play_step()

        if game_over:
            game.show_game_over()
            break

    print(f"Final Score: {score}")

    pygame.quit()


if __name__ == "__main__":
    main()
