import os
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import time

def plot_3d_fit(P, subspace, name):
    frames = []
    assert P.shape[1] == 3, "This visualization only supports 3D data (d=3)."
    k = subspace.shape[1]
    assert k in [1, 2], "Only k=1 or k=2 subspaces are supported for visualization."

    os.makedirs("frames", exist_ok=True)
    os.makedirs("examples", exist_ok=True)
    # Mean for visualization
    mean = np.mean(P, axis=0)

    if k == 1:
        # Plot line
        direction = subspace[:, 0]
        t = np.linspace(-2, 2, 100)
        line = mean + t[:, None] * direction
    elif k == 2:
        # Plot plane
        u, v = subspace[:, 0], subspace[:, 1]
        t = np.linspace(-1, 1, 10)
        T1, T2 = np.meshgrid(t, t)
        X = mean[0] + T1 * u[0] + T2 * v[0]
        Y = mean[1] + T1 * u[1] + T2 * v[1]
        Z = mean[2] + T1 * u[2] + T2 * v[2]
    else:
        raise ValueError("Only subspaces with k = 1, 2, or 3 are supported in 3D.")
    
    # here we make the video.
    for angle in range(0, 360, 5):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if k == 1:
            ax.plot(line[:, 0], line[:, 1], line[:, 2], 'r-', linewidth=2, label='Best-fit line')
        elif k == 2:
            ax.plot_surface(X, Y, Z, alpha=0.5, color='red', label='Best-fit plane')

        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='blue', label='Data points')
        ax.set_title(f"3D Subspace Approximation (k={k})")

        ax.view_init(elev=angle, azim=angle)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        filename = f"frames/frame_{angle:03d}.png"
        plt.savefig(filename, dpi=80)
        plt.close()
        frames.append(imageio.v2.imread(filename))
    imageio.mimsave(f"examples/{name}.gif", frames, duration=0.1)

def plot_2d_fit(P, subspace):
    mean = np.mean(P, axis=0)
    direction = subspace[:, 0]

    t = np.linspace(-2, 2, 100)
    line = mean + t[:, None] * direction

    plt.scatter(P[:, 0], P[:, 1], color='blue', label='Data points')
    plt.plot(line[:, 0], line[:, 1], 'r-', label='Fitted 1D subspace')
    plt.axis('equal')
    plt.legend()
    plt.title("2D Subspace Approximation")
    plt.show()

def optimal_error(P, k):
    pca = PCA(n_components=k).fit(P)
    U = pca.components_.T  # (d, k)
    return projection_error(P, U)

def generate_random_subspace(d):
    A = np.random.randn(d, d)
    S, _ = np.linalg.qr(A)
    return S

def generate_points_on_subspace(S, n):
    d_sub = S.shape[1]
    coefficients = np.random.randn(n, d_sub)  # n points in R^d_sub
    P = coefficients @ S.T                    # shape (n, d)
    return P

def plot_high_dim(n, d):
    result = []
    time_result = []
    S = generate_random_subspace(d)  # Generate random d-dimensional subspace
    P = generate_points_on_subspace(S, n)
    # Center the data
    P -= P.mean(axis=0)
    for k in range(d - 15, d - 1):
        print(f"Testing n={n}, d={d}, k={k}")
        # Run algorithm
        start = time.time()
        U = good_subspace(S, P, k=k, epsilon=0.1)
        t_gs = time.time() - start

        # Compute projection errors
        our = projection_error(P, U)
        
        start = time.time()
        pca = PCA(n_components=k).fit(P)
        t_pca = time.time() - start
        
        pca = optimal_error(P, k)

        print(np.allclose(U.T @ U, np.eye(k), atol=1e-9))
        print("Our: ", our)
        print("PCA: ", pca)
        result.append((abs(our - 10), pca))
        time_result.append((t_gs, t_pca))
        
    # Plot results
    os.makedirs("examples", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(30, d - 1), [r[0] for r in result], label='Our Method', marker='o')
    plt.plot(range(30, d - 1), [r[1] for r in result], label='PCA', marker='x')
    plt.xlabel('Subspace Dimension k')
    plt.ylabel('Projection Error')
    plt.title(f'Projection Error Comparison (n={n}, d={d})')
    plt.legend()
    plt.grid()
    plt.savefig(f"examples/projection_error_{n}_{d}.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(30, d - 1), [r[0] for r in time_result], label='Our Method', marker='o')
    plt.plot(range(30, d - 1), [r[1] for r in time_result], label='PCA', marker='x')
    plt.xlabel('Subspace Dimension k')
    plt.ylabel('Time Comparison')
    plt.title(f'Time Comparison (n={n}, d={d})')
    plt.legend()
    plt.grid()
    plt.savefig(f"examples/Time Comparison_{n}_{d}.png")
    plt.close()
