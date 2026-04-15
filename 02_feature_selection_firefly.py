# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy>=2.0.2",
#     "scikit-learn>=1.6.1",
#     "torch>=2.8.0",
# ]
# ///

# ==========================================
# FEATURE SELECTION - FIREFLY OPTIMIZER
# ==========================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# ==========================================
# FIREFLY CLASS
# ==========================================

class Firefly:
    """Classical firefly for swarm optimization"""
    def __init__(self, dim, bounds=(-0.1, 0.1)):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.brightness = float('inf')
        self.best_position = self.position.copy()
        self.best_brightness = float('inf')
        self.bounds = bounds

    def attract_to(self, other_firefly, attraction=0.5, randomness=0.3, iteration=0, max_iterations=100):
        distance = np.linalg.norm(self.position - other_firefly.position) + 1e-8
        beta0 = 1.0
        gamma = 1.0 / max(1.0, distance)
        beta = beta0 * np.exp(-gamma * distance ** 2)
        attraction_force = beta * (other_firefly.position - self.position)

        # Adaptive parameters
        adaptive_attraction = attraction * (1 - iteration / max(iteration + max_iterations, 1))
        adaptive_randomness = randomness * (1 - iteration / (2 * max_iterations))

        randomization = np.random.uniform(-1, 1, self.position.shape) * adaptive_randomness
        self.position = self.position + attraction_force * adaptive_attraction + randomization
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

# ==========================================
# FIREFLY OPTIMIZER
# ==========================================

class FireflyOptimizer:
    """Classical Firefly Algorithm for feature selection"""
    def __init__(self, model, X_train, y_train, X_val, y_val,
                 n_fireflies=30, max_iterations=100, attraction=0.5,
                 randomness=0.3, batch_size=32,
                 device=torch.device("cpu")):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.device = device

        self.n_fireflies = n_fireflies
        self.max_iterations = max_iterations
        self.attraction = attraction
        self.randomness = randomness
        self.batch_size = batch_size

        self.param_count = sum(p.numel() for p in model.parameters())
        self.fireflies = [Firefly(self.param_count, bounds=(-0.1, 0.1))
                         for _ in range(n_fireflies)]

        self.global_best_position = None
        self.global_best_brightness = float('inf')
        self.criterion = nn.CrossEntropyLoss()
        self.best_fitness_history = []
        self.no_improvement_count = 0

    def evaluate_brightness(self, position):
        try:
            idx = 0
            for param in self.model.parameters():
                param_size = param.numel()
                param.data = torch.tensor(
                    position[idx:idx+param_size].reshape(param.shape),
                    dtype=param.dtype,
                    device=self.device
                )
                idx += param_size

            self.model.eval()
            total_loss = 0
            num_batches = 0

            with torch.no_grad():
                for i in range(0, len(self.X_val), self.batch_size):
                    X_batch = torch.tensor(self.X_val[i:i+self.batch_size], dtype=torch.float32).to(self.device)
                    y_batch = torch.tensor(self.y_val[i:i+self.batch_size], dtype=torch.long).to(self.device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    total_loss += loss.item()
                    num_batches += 1

            return total_loss / max(num_batches, 1)
        except Exception:
            return float('inf')

    def optimize(self, verbose=True, use_hybrid=True):
        if verbose:
            print("\n" + "="*60)
            print("FEATURE SELECTION - FIREFLY OPTIMIZER")
            print("="*60)
            print(f"\nConfiguration:")
            print(f"  Fireflies: {self.n_fireflies}")
            print(f"  Iterations: {self.max_iterations}")
            print(f"  Attraction: {self.attraction}")
            print(f"  Randomness: {self.randomness}\n")

        if verbose:
            print("Initializing firefly population...")
        for firefly in self.fireflies:
            brightness = self.evaluate_brightness(firefly.position)
            firefly.brightness = brightness
            firefly.best_brightness = brightness

            if brightness < self.global_best_brightness:
                self.global_best_brightness = brightness
                self.global_best_position = firefly.position.copy()

        for iteration in range(self.max_iterations):
            sorted_indices = np.argsort([f.brightness for f in self.fireflies])
            sorted_fireflies = [self.fireflies[i] for i in sorted_indices]

            for i, firefly in enumerate(sorted_fireflies):
                for j in range(i):
                    firefly.attract_to(
                        sorted_fireflies[j],
                        attraction=self.attraction,
                        randomness=self.randomness,
                        iteration=iteration,
                        max_iterations=self.max_iterations
                    )

                brightness = self.evaluate_brightness(firefly.position)
                firefly.brightness = brightness

                if brightness < firefly.best_brightness:
                    firefly.best_brightness = brightness
                    firefly.best_position = firefly.position.copy()

                if brightness < self.global_best_brightness:
                    self.global_best_brightness = brightness
                    self.global_best_position = firefly.position.copy()
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

            self.best_fitness_history.append(self.global_best_brightness)

            if verbose and (iteration + 1) % max(1, self.max_iterations // 5) == 0:
                print(f"  Iteration {iteration+1:3d}/{self.max_iterations} | "
                      f"Best Loss: {self.global_best_brightness:.6f}")

            if self.no_improvement_count > 30:
                if verbose:
                    print(f"  Early stopping at iteration {iteration+1}")
                break

        # Apply best weights
        idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = torch.tensor(
                self.global_best_position[idx:idx+param_size].reshape(param.shape),
                dtype=param.dtype,
                device=self.device
            )
            idx += param_size

        # Hybrid refinement with gradient descent
        if use_hybrid:
            if verbose:
                print("\nHybrid Refinement: Fine-tuning with gradient descent...")
            self._hybrid_gradient_refinement(iterations=10, verbose=verbose)

        if verbose:
            print("Optimization Complete!\n" + "="*60 + "\n")

    def _hybrid_gradient_refinement(self, iterations=10, verbose=False):
        """Fine-tune best weights using gradient descent"""
        idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = torch.tensor(
                self.global_best_position[idx:idx+param_size].reshape(param.shape),
                dtype=param.dtype,
                device=self.device
            )
            param.requires_grad = True
            idx += param_size

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for _ in range(iterations):
            self.model.train()
            for j in range(0, len(self.X_train), self.batch_size):
                X_batch = torch.tensor(self.X_train[j:j+self.batch_size], dtype=torch.float32).to(self.device)
                y_batch = torch.tensor(self.y_train[j:j+self.batch_size], dtype=torch.long).to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            val_fitness = self.evaluate_brightness(self._get_weights())
            if val_fitness < self.global_best_brightness:
                self.global_best_brightness = val_fitness
                self.global_best_position = self._get_weights().copy()
                if verbose:
                    print(f"  Refinement improved: {val_fitness:.6f}")

    def _get_weights(self):
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
