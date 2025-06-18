# --- START OF FILE src/deepzero.py ---

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# --- Utility Functions for Parameter Handling ---

def get_flat_params(model):
    """Flattens all model parameters into a single 1D tensor."""
    return torch.cat([p.detach().view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    """Sets model parameters from a flat 1D tensor."""
    offset = 0
    for param in model.parameters():
        param.data.copy_(flat_params[offset:offset + param.numel()].view_as(param))
        offset += param.numel()

def get_param_struct(model):
    """Creates a mapping from flat index to layer/parameter info."""
    param_struct = []
    flat_idx_start = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        param_struct.append({
            'name': name,
            'shape': param.shape,
            'num_params': num_params,
            'start_idx': flat_idx_start,
            'end_idx': flat_idx_start + num_params
        })
        flat_idx_start += num_params
    return param_struct

# --- DeepZero Optimizer Implementation ---

class DeepZero(optim.Optimizer):
    """
    Implements the DeepZero algorithm.

    This optimizer performs Zeroth-Order optimization by estimating gradients
    using a sparse coordinate-wise finite difference approach. It is designed
    for scenarios where first-order gradients are unavailable.
    """

    def __init__(self, model, params, lr=1e-3, mu=1e-3, sparsity_ratio=0.1,
                 p_rge=128, k_sparse=1, momentum=0.9):
        """
        Initializes the DeepZero optimizer.

        Args:
            model (nn.Module): The model to be optimized. Required for forward passes.
            params (iterable): Iterable of parameters to optimize.
            lr (float): Learning rate.
            mu (float): Perturbation value for finite differences.
            sparsity_ratio (float): The fraction of parameters to keep active (1.0 - pruning ratio).
            p_rge (int): Number of random query directions for RGE during ZO-GraSP.
            k_sparse (int): Frequency (in epochs) to update the active sparse set.
            momentum (float): Momentum factor for the SGD update rule.
        """
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < mu:
            raise ValueError(f"Invalid mu value: {mu}")
        if not 0.0 < sparsity_ratio <= 1.0:
            raise ValueError(f"Invalid sparsity_ratio: {sparsity_ratio}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        self.model = model
        self.device = next(model.parameters()).device

        defaults = dict(lr=lr, mu=mu, sparsity_ratio=sparsity_ratio,
                        p_rge=p_rge, k_sparse=k_sparse, momentum=momentum)
        super(DeepZero, self).__init__(params, defaults)

        # State initialization
        self.param_groups[0]['params'] = list(self.model.parameters())  # Ensure all params are here
        self.param_struct = get_param_struct(self.model)
        self.total_params = sum(p['num_params'] for p in self.param_struct)

        self.lprs = None
        self.active_indices = None
        self.epoch_counter = 0
        self.iter_counter = 0

        print("--- Initializing DeepZero Optimizer ---")
        self._initialize_sparsity()
        self._update_active_set()
        print("--- DeepZero Initialization Complete ---")

    def _rge_gradient_estimate(self, closure, p_rge):
        """
        Estimates gradient using Randomized Gradient-free Estimation (RGE).
        Used internally for the ZO-GraSP procedure.
        """
        mu = self.param_groups[0]['mu']

        original_params = get_flat_params(self.model)
        grad_estimate = torch.zeros_like(original_params)

        # Baseline loss
        l_base = closure().item()

        for _ in range(p_rge):
            u = torch.randn_like(original_params)
            u /= torch.norm(u)  # Normalize the random direction vector

            # Perturb and calculate loss
            set_flat_params(self.model, original_params + mu * u)
            l_plus = closure().item()

            # Accumulate gradient component
            grad_estimate += ((l_plus - l_base) / mu) * u

        # Restore original parameters and average
        set_flat_params(self.model, original_params)
        return grad_estimate / p_rge

    def _initialize_sparsity(self):
        """
        Performs the one-time ZO-GraSP procedure to determine Layer-wise Pruning Ratios (LPRs).
        """
        print("Starting ZO-GraSP to determine sparsity structure...")
        p_rge = self.param_groups[0]['p_rge']
        mu = self.param_groups[0]['mu']
        sparsity_ratio = self.param_groups[0]['sparsity_ratio']

        # A dummy closure for initialization on a single batch (or random data)
        # In a real scenario, you might want to use a representative batch of data.
        dummy_input = torch.randn(2, 3, 32, 32, device=self.device)
        dummy_target = torch.randint(0, 10, (2,), device=self.device)

        def init_closure():
            self.model.zero_grad()
            output = self.model(dummy_input)
            return F.cross_entropy(output, dummy_target)

        original_params = get_flat_params(self.model)

        # Step 1: Estimate initial gradient g_hat
        print("  (1/4) Estimating initial gradient (g_hat)...")
        g_hat = self._rge_gradient_estimate(init_closure, p_rge)

        # Step 2: Estimate gradient at perturbed point
        print("  (2/4) Estimating perturbed gradient...")
        set_flat_params(self.model, original_params + mu * g_hat)
        g_hat_perturbed = self._rge_gradient_estimate(init_closure, p_rge)
        set_flat_params(self.model, original_params)  # Restore

        # Step 3: Approximate Hessian-gradient product and calculate scores
        print("  (3/4) Calculating importance scores...")
        Hg_approx = (g_hat_perturbed - g_hat) / mu
        scores = -original_params * Hg_approx

        # Step 4: Determine LPRs from scores
        print("  (4/4) Determining Layer-wise Pruning Ratios (LPRs)...")
        num_to_keep = int(self.total_params * sparsity_ratio)

        # Find global threshold
        threshold = torch.kthvalue(torch.abs(scores), self.total_params - num_to_keep).values

        self.lprs = {}
        total_kept = 0
        for layer_info in self.param_struct:
            layer_scores = torch.abs(scores[layer_info['start_idx']:layer_info['end_idx']])
            kept_mask = layer_scores >= threshold

            num_kept_in_layer = kept_mask.sum().item()
            ratio = num_kept_in_layer / layer_info['num_params']
            self.lprs[layer_info['name']] = ratio
            total_kept += num_kept_in_layer

        print(f"ZO-GraSP finished. Target sparsity: {1 - sparsity_ratio:.2%}. "
              f"Actual sparsity: {1 - total_kept / self.total_params:.2%}")

    def _update_active_set(self):
        """
        Updates the set of active indices (S_t) based on the fixed LPRs.
        This implements the dynamic sparsity pattern.
        """
        print(f"\nEpoch {self.epoch_counter}: Updating active sparse set...")
        active_indices = []
        for layer_info in self.param_struct:
            layer_name = layer_info['name']
            num_params_in_layer = layer_info['num_params']
            start_idx = layer_info['start_idx']

            ratio = self.lprs[layer_name]
            num_to_keep = int(np.round(num_params_in_layer * ratio))

            # Randomly sample indices from this layer's range
            layer_indices = np.arange(start_idx, start_idx + num_params_in_layer)
            chosen_indices = np.random.choice(layer_indices, num_to_keep, replace=False)
            active_indices.extend(chosen_indices)

        self.active_indices = torch.tensor(sorted(active_indices), device=self.device, dtype=torch.long)
        print(f"New active set size: {len(self.active_indices)} ({len(self.active_indices) / self.total_params:.2%})")

    def step(self, closure):
        """
        Performs a single optimization step.

        Args:
            closure (callable): A closure that re-evaluates the model and returns the loss.
        """
        # Check if it's time to update the active set
        if self.iter_counter == 0 and self.epoch_counter > 0 and \
                self.epoch_counter % self.param_groups[0]['k_sparse'] == 0:
            self._update_active_set()

        loss = self._sparse_cge_gradient_estimate(closure)
        self.iter_counter += 1
        return loss

    def _sparse_cge_gradient_estimate(self, closure):
        """
        Estimates gradient using Sparse Coordinate-wise Gradient Estimation (Sparse-CGE).
        This is the main workhorse of the optimizer.
        """
        mu = self.param_groups[0]['mu']
        lr = self.param_groups[0]['lr']
        momentum = self.param_groups[0]['momentum']

        original_params = get_flat_params(self.model)
        grad_estimate = torch.zeros_like(original_params)

        # 1. Calculate baseline loss
        l_base = closure().item()

        # 2. Iterate through the ACTIVE indices only
        for j in self.active_indices:
            # Perturb one parameter
            original_val = original_params[j].item()
            original_params[j] += mu
            set_flat_params(self.model, original_params)

            # Calculate perturbed loss
            l_plus = closure().item()

            # Restore parameter
            original_params[j] = original_val
            set_flat_params(self.model, original_params)

            # Calculate and store partial derivative estimate
            grad_estimate[j] = (l_plus - l_base) / mu

        # 3. Perform the parameter update using the sparse gradient
        for i, p in enumerate(self.model.parameters()):
            param_state = self.state[p]

            # Un-flatten the gradient for this parameter
            layer_info = self.param_struct[i]
            param_grad = grad_estimate[layer_info['start_idx']:layer_info['end_idx']].view_as(p)

            if 'momentum_buffer' not in param_state:
                param_state['momentum_buffer'] = torch.clone(param_grad).detach()
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(param_grad)

            p.data.add_(param_state['momentum_buffer'], alpha=-lr)

        return l_base

# --- END OF FILE src/deepzero.py ---
