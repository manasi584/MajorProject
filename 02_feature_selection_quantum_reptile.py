# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy>=2.0.2",
#     "scikit-learn>=1.6.1",
#     "torch>=2.8.0",
# ]
# ///

# ==========================================
# FEATURE SELECTION - QUANTUM REPTILE OPTIMIZER
# ==========================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# ==========================================
# QUANTUM REPTILE CLASS
# ==========================================

class QuantumReptile:
    """Quantum-mutated reptile for crocodile-inspired optimization"""
    def __init__(self, dim, bounds=(-0.1, 0.1)):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.energy = float('inf')
        self.best_position = self.position.copy()
        self.best_energy = float('inf')
        self.bounds = bounds
        self.quantum_phase = np.random.uniform(0, 2*np.pi, dim)
        self.qbit_alpha = np.random.uniform(0, 1, dim)
        self.qbit_beta = np.sqrt(1 - self.qbit_alpha**2)

    def update_quantum_mutation(self, step_size=0.1):
        self.quantum_phase += np.random.uniform(-step_size, step_size, self.quantum_phase.shape)
        self.quantum_phase = self.quantum_phase % (2 * np.pi)
        rotation_angle = np.random.uniform(0, 2*np.pi, self.qbit_alpha.shape)
        self.qbit_alpha = np.cos(rotation_angle)
        self.qbit_beta = np.sin(rotation_angle)

    def encircle_prey(self, prey_position, encircle_factor=0.5):
        coeff = 2 * encircle_factor * np.random.rand(*self.position.shape) - encircle_factor
        self.position = prey_position - coeff * (prey_position - self.position)

    def hunt_cooperatively(self, pack_positions, hunt_factor=0.3):
        pack_center = np.mean(pack_positions, axis=0) if len(pack_positions) > 0 else self.position
        hunt_coeff = hunt_factor * np.random.rand(*self.position.shape)
        self.position = self.position + hunt_coeff * (pack_center - self.position)

    def apply_quantum_mutation(self, mutation_rate=0.1, iteration=0, max_iterations=100):
        adaptive_mutation_rate = mutation_rate * (1 - iteration / max(iteration + max_iterations, 1))
        if np.random.rand() < adaptive_mutation_rate:
            quantum_mutation = (self.qbit_alpha * np.cos(self.quantum_phase) +
                                self.qbit_beta * np.sin(self.quantum_phase))
            mutation_intensity = (1 - iteration / max(iteration + max_iterations, 1)) * 0.05
            self.position = self.position + mutation_intensity * quantum_mutation
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        self.update_quantum_mutation()

# ==========================================
# QUANTUM REPTILE OPTIMIZER
# ==========================================

class QuantumMutationReptileOptimizer:
    """Quantum Mutation Reptile Search Algorithm"""
    def __init__(self, model, X_train, y_train, X_val, y_val,
                 n_reptiles=30, max_iterations=100, encircle_factor=0.5,
                 hunt_factor=0.3, mutation_rate=0.1, batch_size=32,
                 device=torch.device("cpu")):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.device = device

        self.n_reptiles = n_reptiles
        self.max_iterations = max_iterations
        self.encircle_factor = encircle_factor
        self.hunt_factor = hunt_factor
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size

        self.param_count = sum(p.numel() for p in model.parameters())
        self.reptiles = [QuantumReptile(self.param_count, bounds=(-0.1, 0.1))
                        for _ in range(n_reptiles)]

        self.global_best_position = None
        self.global_best_energy = float('inf')
        self.criterion = nn.CrossEntropyLoss()
        self.best_fitness_history = []
        self.no_improvement_count = 0

    def evaluate_energy(self, position):
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
            print("FEATURE SELECTION - QUANTUM REPTILE OPTIMIZER")
            print("="*60)
            print(f"\nConfiguration:")
            print(f"  Population: {self.n_reptiles} reptiles")
            print(f"  Iterations: {self.max_iterations}")
            print(f"  Encircle Factor: {self.encircle_factor}")
            print(f"  Hunt Factor: {self.hunt_factor}")
            print(f"  Mutation Rate: {self.mutation_rate}\n")

        if verbose:
            print("Initializing reptile population...")
        for reptile in self.reptiles:
            energy = self.evaluate_energy(reptile.position)
            reptile.energy = energy
            reptile.best_energy = energy
            if energy < self.global_best_energy:
                self.global_best_energy = energy
                self.global_best_position = reptile.position.copy()

        for iteration in range(self.max_iterations):
            adaptive_encircle = self.encircle_factor * (1 - iteration / self.max_iterations)
            adaptive_hunt = self.hunt_factor * (1 - 0.5 * iteration / self.max_iterations)
            adaptive_mutation = self.mutation_rate * (1 - iteration / self.max_iterations)

            sorted_indices = np.argsort([r.energy for r in self.reptiles])
            alpha_reptile = self.reptiles[sorted_indices[0]]
            beta_reptile = self.reptiles[sorted_indices[1]] if len(self.reptiles) > 1 else alpha_reptile

            for i, reptile in enumerate(self.reptiles):
                if np.random.rand() < 0.5:
                    reptile.encircle_prey(alpha_reptile.position, adaptive_encircle)
                else:
                    reptile.encircle_prey(beta_reptile.position, adaptive_encircle)

                nearby_idx = np.random.choice(len(self.reptiles),
                                              size=max(1, len(self.reptiles)//3),
                                              replace=False)
                nearby_positions = [self.reptiles[idx].position for idx in nearby_idx]
                reptile.hunt_cooperatively(nearby_positions, adaptive_hunt)

                reptile.apply_quantum_mutation(
                    mutation_rate=adaptive_mutation,
                    iteration=iteration,
                    max_iterations=self.max_iterations
                )

                energy = self.evaluate_energy(reptile.position)
                reptile.energy = energy

                if energy < reptile.best_energy:
                    reptile.best_energy = energy
                    reptile.best_position = reptile.position.copy()

                if energy < self.global_best_energy:
                    self.global_best_energy = energy
                    self.global_best_position = reptile.position.copy()
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

            self.best_fitness_history.append(self.global_best_energy)

            if verbose and (iteration + 1) % max(1, self.max_iterations // 5) == 0:
                print(f"  Iteration {iteration+1:3d}/{self.max_iterations} | "
                      f"Best Loss: {self.global_best_energy:.6f}")

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

            val_fitness = self.evaluate_energy(self._get_weights())
            if val_fitness < self.global_best_energy:
                self.global_best_energy = val_fitness
                self.global_best_position = self._get_weights().copy()
                if verbose:
                    print(f"  Refinement improved: {val_fitness:.6f}")

    def _get_weights(self):
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
