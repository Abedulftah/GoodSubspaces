import numpy as np
from scipy.linalg import null_space

def orthogonal_complement_in_subspace(line_vec, subspace_basis):
    """
    Compute the orthogonal complement of a line (given by a vector)
    inside a subspace (given by its basis vectors).
    
    Parameters:
        line_vec (ndarray): shape (n,), the direction vector of the line (must be in the subspace)
        subspace_basis (ndarray): shape (n, k), the basis of the subspace
    
    Returns:
        ndarray: shape (n, m), a basis of the orthogonal complement of the line in the subspace
    """
    # Ensure column vectors
    v = line_vec.reshape(-1, 1)  # shape (k, 1)
    B = subspace_basis           # shape (n, k)

    # Verify v is in the subspace (optional but recommended)
    coeffs, _, _, _ = np.linalg.lstsq(B, v, rcond=None)
    if not np.allclose(B @ coeffs, v):
        raise ValueError("line_vec must lie in the subspace")

    # Constraint: v.T @ (B @ c) = 0  -->  (v.T @ B) @ c = 0
    A = v.T @ B  # shape (1, k)

    # Find null space of A (coefficients for orthogonal complement)
    complement_coeffs = null_space(A)  # shape (k, m), where m = k-1 (if v is in S)

    # Orthogonal complement basis: B @ complement_coeffs
    orth_complement = B @ complement_coeffs  # shape (n, m)

    return orth_complement

def good_subspace(S, P, k, epsilon):
    """
    Approximate best-fit k-dimensional subspace using randomized sampling.
    
    Implements the recursive Good-Subspace algorithm from:
    "Efficient Subspace Approximation Algorithms" by Shyamalkumar and Varadarajan.

    Parameters:
    -----------
    P : ndarray of shape (n, d)
        A set of n points in d-dimensional Euclidean space.
    k : int
        Target subspace dimension (1 <= k < d).
    epsilon : float
        Approximation parameter controlling error vs. runtime (0 < epsilon < 1).

    Returns:
    --------
    U : ndarray of shape (d, k)
        An orthonormal basis for a k-dimensional subspace approximating the best-fit flat.
    """
    n, d = P.shape

    # Compute norms of all points from origin (used for importance sampling)
    P_norms = np.linalg.norm(P, axis=1)
    total_norm = np.sum(P_norms)
    probs = P_norms / total_norm if total_norm > 1e-8 else np.ones(n) / n


    # If all points are zero, return any arbitrary k-subspace (e.g., first k identity vectors)
    if np.all(P == 0):
        print("All points are zero, returning trivial subspace.")
        return np.eye(d)[:, :k]

    # Step 1: Generate a sequence of i candidate lines
    c = 20  # Tunable constant controlling success probability (paper uses "c > 0")
    i = int(np.ceil((c / epsilon) * np.log(1 / epsilon)))  # Number of candidate lines
    lines = []

    # l_0
    p = 0
    while np.linalg.norm(p) == 0:
        p = P[np.random.choice(n, p=probs)]
    norm_p = np.linalg.norm(p)


    lines.append(p / norm_p)

    while len(lines) < i:
        
        r = P[np.random.choice(n, p=probs)]
        norm_r = np.linalg.norm(r)
        if norm_r == 0:
            continue

        u = lines[-1]
        v = r / norm_r

        alpha = np.random.uniform(0, 1)
        seg = (1 - alpha) * u + alpha * (v if np.random.rand() < 0.5 else -v)

        norm_seg = np.linalg.norm(seg)
        if norm_seg > 0:
            lines.append(seg / norm_seg)

    # Step 3: Uniformly choose one line ℓ from the generated list
    chosen_line = lines[np.random.randint(len(lines))].reshape(-1, 1)
    # Base case: k == 1 → return 1D subspace (just the line)
    if k == 1:
        print(f"Returning 1D subspace: {chosen_line.shape}")
        return chosen_line

    # Step 4: Project P onto orthogonal complement of chosen_line
    # This is S' ← S ⊥ ℓ
    S_orth = orthogonal_complement_in_subspace(chosen_line, S)  # New basis spanning S ⊥ ℓ
    # print(f"Orthogonal subspace shape: {S_orth.shape}, Chosen line shape: {chosen_line.shape}")
    # Project P onto this orthogonal subspace
    P_proj = S_orth @ (S_orth.T @ P.T) # Project all points onto S ⊥ ℓ
    
    # Step 5: Recursive call to find (k-1)-subspace in projected space
    G = good_subspace(S_orth, P_proj.T, k - 1, epsilon)
    # Step 6: Lift the result back to original space and span with chosen_line
    # Final subspace is span(ℓ ∪ G)
    U = np.hstack((chosen_line, G))
    return U
