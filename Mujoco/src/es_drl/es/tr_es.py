import os
import numpy as np

# import torch TODO: CHANGE TO JAX
# from torch import nn TODO: CHANGE TO JAX
from joblib import Parallel, delayed

from src.es_drl.es.basic_es import BasicES


class TrustRegionES(BasicES):
    def __init__(self, common_cfg, es_cfg):
        super().__init__(common_cfg, es_cfg)

        # TRES-specific parameters
        self.epochs_per_iter = es_cfg.get("epochs", 10)
        self.clip_ratio = es_cfg.get("clip_ratio", 0.2)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def _compute_surrogate_loss(self, old_params_np, rewards, noise):
        """
        PPO-style clipped surrogate objective for TRES.
        """
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        old_params = torch.tensor(
            old_params_np, dtype=torch.float32, device=self.device
        )
        losses = []
        for i in range(self.population_size):
            candidate_params = old_params + self.sigma * noise[i]
            self._set_param_vector(candidate_params.detach().cpu().numpy())
            new_vec = self._get_param_vector()
            new_vec = torch.tensor(
                new_vec, dtype=torch.float32, device=self.device, requires_grad=True
            )

            # KL-based ratio proxy
            ratio = torch.exp(
                -0.5 * torch.sum((new_vec - old_params.detach()) ** 2) / (self.sigma**2)
            )
            clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            loss = -torch.min(ratio * rewards[i], clipped * rewards[i])
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        return total_loss

    def run(self) -> str:
        mu = torch.from_numpy(self._get_param_vector()).to(self.device)
        param_dim = mu.numel()

        for gen in range(self.num_generations):
            if self.verbose:
                print(f"[TRES] GENERATION {gen} START")

            noise = torch.randn(self.population_size, param_dim, device=self.device)
            candidates = (mu.unsqueeze(0) + self.sigma * noise).cpu().numpy()
            rewards = np.array(
                Parallel(n_jobs=-1)(
                    delayed(self._evaluate_candidate)(cand) for cand in candidates
                )
            ).astype(np.float32)

            # Gradient-based refinement using sampled data
            for _ in range(self.epochs_per_iter):
                self.optimizer.zero_grad()
                loss = self._compute_surrogate_loss(mu.cpu().numpy(), rewards, noise)
                loss.backward()
                self.optimizer.step()
                mu = torch.from_numpy(self._get_param_vector()).to(self.device)

            if self.verbose:
                print(f"[TRES] Reward Mean: {np.mean(rewards):.2f}")

        self._set_param_vector(mu.cpu().numpy())
        ckpt_path = os.path.join(
            self.model_dir, f"{self.es_name}_TRES_seed{self.seed}.pt"
        )
        torch.save(self.policy.state_dict(), ckpt_path)
        return ckpt_path
