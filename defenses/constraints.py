import math
import numpy as np
import torch

def assert_dm(rho):
    """
    Assert that the input tensor is a valid batch of density matrices.
    Args:
        rho (torch.Tensor): Tensor of shape [batch_size, m, m].
    """
    # Ensure the tensor is Hermitian
    assert torch.allclose(rho, rho.transpose(-1, -2).conj()), "Density matrices must be Hermitian."

    # Ensure the tensor is positive semi-definite
    eigvals = torch.linalg.eigvalsh(rho)  # Batched eigenvalue computation
    assert torch.all(eigvals >= -1e-6), "Density matrices must be positive semi-definite (allowing small numerical errors)."

    # Ensure the tensor has unit trace
    traces = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)
    assert torch.allclose(traces, torch.ones_like(traces), atol=1e-6), "Density matrices must have unit trace."
    
def distance_between_matrices(A, B):
    """
    Compute the relative Frobenius distance between matrices A (adversarial) and B (enforced DM).

    Parameters:
    A (torch.Tensor): Arbitrary adversarial matrix (batch of matrices).
    B (torch.Tensor): Enforced density matrix (batch of matrices with unit trace).

    Returns:
    torch.Tensor: Relative Frobenius distances for each matrix in the batch.
    """
    # Compute the Frobenius norm of the difference
    difference = A - B
    frobenius_diff = torch.norm(difference, dim=(-2, -1), p="fro")
    return frobenius_diff


def enforce_dm_conditions(batch_rho):
    """
    Enforce the density matrix conditions with minimal adjustments and optimized performance:
    1. Hermitian symmetry
    2. Positive semi-definiteness
    3. Trace = 1
    
    Args:
        rho (Tensor): Flattened density matrices of shape [batch_size, m^2].
        matrix_size (int): The size of the original density matrix (m).
        
    Returns:
        Tensor: Flattened density matrices after constraint enforcement.
    """

    batch_size = batch_rho.size(0)  # Batch size
    m_squared = batch_rho.size(1)  # Should be m^2
    matrix_size = int(m_squared ** 0.5)
    assert matrix_size * matrix_size == m_squared, "Flattened dimension is not a perfect square."

    # Reshape each vector back into its original m x m matrix form
    rho_matrices_original = batch_rho.view(batch_size, matrix_size, matrix_size)

    # Ensure Hermitian symmetry
    rho_matrices = 0.5 * (rho_matrices_original + rho_matrices_original.transpose(-1, -2).conj())

    # Eigenvalue decomposition (batched if possible)
    eigvals, eigvecs = torch.linalg.eigh(rho_matrices)  # Efficient for Hermitian matrices

    # Ensure positive semi-definiteness
    eigvals.clamp_(min=0)

    # Reconstruct the positive semi-definite matrix
    rho_matrices = (eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.transpose(-1, -2).conj()

    # Normalize trace to 1
    trace = torch.diagonal(rho_matrices, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    rho_matrices = rho_matrices / trace
    
    # assert_dm(rho_matrices)
    # distance_between_matrices(rho_matrices_original, rho_matrices)
    
    # Flatten the matrices back to their original form
    rho_flattened = rho_matrices.view(batch_size, -1)

    return rho_flattened


def assert_stats(min_vals, max_vals, median_vals, mean_vals, var_vals):
    """
    Assert that the input tensor satisfies the statistical constraints:
    [min_f1, ..., min_fm, max_f1, ..., max_fm, median_f1, ..., mean_fm, var_f1, ..., var_fm].

    Args:
        stats_tensor (torch.Tensor): Input tensor of shape [batch_size, 155].
    """
    assert torch.all(min_vals <= max_vals), "Minimum values must be less than or equal to maximum values."
    assert torch.all((median_vals >= min_vals) & (median_vals <= max_vals)), "Median values must lie between min and max."
    assert torch.all((mean_vals >= min_vals) & (mean_vals <= max_vals)), "Mean values must lie between min and max."
    assert torch.all(var_vals >= 0), "Variance values must be non-negative."


def distance_between_stats(stats_adversarial, stats_enforced):
    """
    Compute the relative Frobenius distance between adversarial stats and enforced stats.

    Parameters:
    stats_adversarial (torch.Tensor): Adversarial statistics tensor of shape [batch_size, 155].
    stats_enforced (torch.Tensor): Enforced statistics tensor of shape [batch_size, 155].

    Returns:
    torch.Tensor: Relative Frobenius distances for each batch.
    """
    # Compute the Frobenius norm of the difference
    difference = stats_adversarial - stats_enforced
    frobenius_diff = torch.norm(difference, dim=1, p="fro")  # Frobenius norm for each batch element
    return frobenius_diff


def enforce_stats_conditions(stats_tensor):
    """
    Enforce statistical constraints on a tensor with minimal adjustments.

    Args:
        stats_tensor (torch.Tensor): Input tensor of shape [batch_size, 155].

    Returns:
        torch.Tensor: Corrected tensor with enforced constraints.
    """
        
    batch_size, num_elements = stats_tensor.shape
    num_features = num_elements // 5

    stats_tensor_reshaped = stats_tensor.view(batch_size, num_features, 5)
    min_vals = stats_tensor_reshaped[:, :, 0].clone()
    max_vals = stats_tensor_reshaped[:, :, 1].clone()
    median_vals = stats_tensor_reshaped[:, :, 2].clone()
    mean_vals = stats_tensor_reshaped[:, :, 3].clone()
    var_vals = stats_tensor_reshaped[:, :, 4].clone()

    # Enforce variance ≥ 0
    var_neg_mask = var_vals < 0
    var_vals[var_neg_mask] = 0.0  # Minimal change to zero

    # Enforce min ≤ max
    min_gt_max_mask = min_vals > max_vals
    min_adjust = min_gt_max_mask & ((min_vals - max_vals) < (max_vals - min_vals))
    max_adjust = min_gt_max_mask & ~min_adjust
    min_vals[min_adjust] = max_vals[min_adjust]  # Adjust min to max where change is smaller
    max_vals[max_adjust] = min_vals[max_adjust]  # Adjust max to min where change is smaller

    # Enforce mean within [min, max]
    mean_below_min = mean_vals < min_vals
    mean_above_max = mean_vals > max_vals
    mean_vals[mean_below_min] = min_vals[mean_below_min]  # Adjust mean to min
    mean_vals[mean_above_max] = max_vals[mean_above_max]  # Adjust mean to max

    # Enforce median within [min, max]
    median_below_min = median_vals < min_vals
    median_above_max = median_vals > max_vals
    median_vals[median_below_min] = min_vals[median_below_min]  # Adjust median to min
    median_vals[median_above_max] = max_vals[median_above_max]  # Adjust median to max

    # Reassemble the tensor
    corrected_stats_reshaped = torch.stack([min_vals, max_vals, median_vals, mean_vals, var_vals], dim=2)
    corrected_stats = corrected_stats_reshaped.view(batch_size, -1)
    
    # assert_stats(min_vals, max_vals, median_vals, mean_vals, var_vals)
    # distance_between_stats(stats_tensor, corrected_stats)

    return corrected_stats
