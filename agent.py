from typing import final
import torch
import random
import numpy as np
from collections import deque

from game import SnakeGameAI, Point, Direction, BLOCK_SIZE
from model import Linear_QNet, QTrainer
import graph

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:
    def __init__(self) -> None:
        self.n_games: int = 0
        self.epsilon: float = 0  # Randomness
        self.gamma: float = 0.9  # Discount rate
        self.memory: deque = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model: Linear_QNet = Linear_QNet(
            input_dim=11, hidden_dim=256, output_dim=3
        )  # input= Size of the state, output= size of the action eg.[1, 0, 0]
        for param in self.model.parameters():
            print(param)
        if self.model.load:
            self.model.load_state_dict(self.model.load())
            self.model.eval()
            print("LOADED")
            for param in self.model.parameters():
                print(param)

        self.trainer: QTrainer = QTrainer(
            self.model, lr=LEARNING_RATE, gamma=self.gamma
        )

    def get_state(self, game: SnakeGameAI):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food left
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y,  # Food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff exploration / exploration
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0: torch.Tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores, plot_mean_scores = [], []
    total_score: int = 0
    record: int = 0

    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get old state
        old_state = agent.get_state(game)

        # Get move
        final_move = agent.get_action(old_state)

        # Perform move and get new state
        reward, done, score = game.play_step(action=final_move)
        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(old_state, final_move, reward, new_state, done)

        # REmember
        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
            # Train long memory (Experience replay)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save(file_name="model2.pth")

            print(f"Game: {agent.n_games}, Score: {score}, Record: {record}")

            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.n_games)
            graph.plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
