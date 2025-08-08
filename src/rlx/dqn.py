import mlx.core as mx
import mlx.nn as nn
import numpy as np

class DQN:
    def __init__(self, q_network, target_network, optimizer):
        self.q_network = q_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.loss_and_grad_fn = nn.value_and_grad(self.q_network, self.loss_fn)

    def loss_fn(self, td_target, observations, actions):
        """Compute DQN loss with proper tensor indexing"""
        # Get Q-values for all actions
        q_values = self.q_network(observations)  # Shape: (batch_size, num_actions)
        
        # Convert actions to proper shape for indexing
        actions = actions.astype(mx.int32)  # Ensure int32 type
        
        # Get Q-values for the selected actions using advanced indexing
        batch_indices = mx.arange(len(actions))
        current_q_values = q_values[batch_indices, actions]  # Shape: (batch_size,)
        
        # Compute MSE loss
        loss = mx.mean((current_q_values - td_target) ** 2)
        return loss

    def update(self, data, args):
        """Update the DQN with proper tensor handling"""
        # Convert data to MLX arrays with consistent dtypes
        observations = mx.array(data.observations, dtype=mx.float32)
        next_observations = mx.array(data.next_observations, dtype=mx.float32)
        actions = mx.array(data.actions.flatten(), dtype=mx.int32)
        rewards = mx.array(data.rewards.flatten(), dtype=mx.float32)
        dones = mx.array(data.dones.flatten(), dtype=mx.float32)
        
        # Compute target Q-values (no gradients needed for target network)
        next_q_values = self.target_network(next_observations)
        target_max = mx.max(next_q_values, axis=1)
        td_target = rewards + args.gamma * target_max * (1 - dones)
        
        # Compute loss and gradients
        loss, grads = self.loss_and_grad_fn(td_target, observations, actions)
        
        # Update network
        self.optimizer.update(self.q_network, grads)
        mx.eval(self.q_network.parameters(), self.optimizer.state)
        
        return float(loss)
