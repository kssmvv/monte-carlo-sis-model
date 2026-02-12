"""
SIS Epidemic Simulation on Scale-Free Networks

Simulates the spread of infectious diseases on Barabasi-Albert scale-free
networks using the SIS (Susceptible-Infected-Susceptible) model. Generates
networks via preferential attachment, runs Monte Carlo simulations of disease
dynamics, and produces analysis plots showing how infection/recovery rates
and network topology affect epidemic outcomes.

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from colorama import Fore
from itertools import product


# ---------------------------------------------------------------------------
# Network construction (Barabasi-Albert model)
# ---------------------------------------------------------------------------

def full_mesh(n_nodes):
    """Create a fully connected adjacency matrix for n_nodes nodes."""
    return np.ones((n_nodes, n_nodes)) - np.eye(n_nodes)


def compute_p(G):
    """Compute preferential attachment probabilities from node degrees."""
    degree_sum = np.sum(G, axis=0)
    return degree_sum / np.sum(degree_sum)


def select_attachment_nodes(G, m):
    """Select m nodes for preferential attachment, weighted by degree."""
    p = compute_p(G)
    return np.random.choice(G.shape[0], size=m, p=p)


def BA_model(initial_size, m, final_size):
    """
    Build a network using the Barabasi-Albert preferential attachment model.

    Starts from a fully connected seed graph of `initial_size` nodes and
    iteratively adds new nodes, each connecting to `m` existing nodes chosen
    with probability proportional to their degree, until the network reaches
    `final_size` nodes.
    """
    G = full_mesh(initial_size)
    for i in range(initial_size, final_size):
        G = np.pad(G, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        targets = select_attachment_nodes(G, m)
        for node in targets:
            G[node, i] = 1
            G[i, node] = 1
    return G


def normalize_out_degree(G):
    """Normalize each row of the adjacency matrix by the node's out-degree."""
    row_sums = np.sum(G, axis=1) - np.diag(G)
    row_sums[row_sums == 0] = 1  # avoid division by zero for isolated nodes
    return G / row_sums[:, np.newaxis]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def print_matrix(G):
    """Print the adjacency matrix with color-coded entries (green = edge)."""
    for i, j in product(range(G.shape[0]), range(G.shape[1])):
        ending = "\n" if j == G.shape[0] - 1 else ""
        if G[i, j] == 0:
            print(Fore.WHITE, f'{round(G[i, j], 2)}', end=ending)
        if G[i, j] > 0:
            print(Fore.GREEN, f'{round(G[i, j], 2)}', end=ending)


def plot_network(G):
    """Draw the network graph using a spring layout."""
    graph = nx.from_numpy_array(G, create_using=nx.DiGraph())
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=100, font_size=5)
    plt.show()


def plot_degree_histogram(G):
    """Plot a histogram of node degrees in the network."""
    degrees = np.sum(G, axis=0) + np.sum(G, axis=1) - np.diag(G)
    unique_degrees, counts = np.unique(degrees, return_counts=True)

    plt.figure(figsize=(10, 6))
    plt.bar(unique_degrees, counts, color='skyblue')
    plt.xlabel('Node degree')
    plt.ylabel('Number of nodes')
    plt.title('Degree Distribution')
    plt.grid(True)
    plt.xticks(unique_degrees)
    plt.tight_layout()
    plt.show()


def plot_network_colored(G, nodes):
    """Draw the network with infected nodes in red and susceptible in blue."""
    graph = nx.from_numpy_array(G, create_using=nx.DiGraph())
    susceptible = np.where(nodes == "s")[0]
    infected = np.where(nodes == "i")[0]
    pos = nx.spring_layout(graph)

    plt.figure(figsize=(12, 12))
    plt.axis('off')
    nx.draw_networkx_nodes(graph, pos, nodelist=susceptible, node_color='#A0CBE2')
    nx.draw_networkx_nodes(graph, pos, nodelist=infected, node_color='red')
    nx.draw_networkx_edges(graph, pos, alpha=0.4)
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='verdana')
    plt.show()


# ---------------------------------------------------------------------------
# SIS epidemic model
# ---------------------------------------------------------------------------

def initial_infection(n_nodes, initial_infected_count=None):
    """
    Create an initial infection state array.

    Each node is either 's' (susceptible) or 'i' (infected). If no count is
    given, a random number of nodes are infected.
    """
    nodes = np.full(n_nodes, "s")
    if initial_infected_count is None:
        initial_infected_count = np.random.randint(1, n_nodes + 1)
    infected_indices = np.random.choice(n_nodes, initial_infected_count, replace=False)
    nodes[infected_indices] = "i"
    return nodes


def SIS_step(G, nodes, mu, beta):
    """
    Compute the infection probability for each node in the next time step.

    For each node i, the probability of being infected is derived from:
    - the product of (1 - beta * pi_j * G[j,i]) over all neighbors j,
      giving the probability of *not* being infected by any neighbor (qi),
    - the recovery rate mu for currently infected nodes.

    The update rule accounts for susceptible nodes getting infected,
    infected nodes staying infected, and infected nodes recovering but
    getting re-infected in the same step.
    """
    new_prob = np.zeros(len(nodes))
    pi = np.where(nodes == 'i', 1, 0)

    for i in range(len(G)):
        qi = 1
        for j in range(len(G)):
            qi *= (1 - beta * pi[j] * G[j, i])
        new_prob[i] = (1 - qi) * (1 - pi[i]) + (1 - mu) * pi[i] + mu * pi[i] * (1 - qi)

    return new_prob


def new_state_nodes(prob, nodes):
    """
    Update each node's state based on its infection probability.

    Draws a uniform random number for each node; if the probability exceeds
    the draw, the node becomes (or stays) infected, otherwise susceptible.
    """
    for i in range(len(prob)):
        if prob[i] > np.random.uniform(0, 1):
            nodes[i] = 'i'
        else:
            nodes[i] = 's'


# ---------------------------------------------------------------------------
# Analysis plots
# ---------------------------------------------------------------------------

def plot_infection_evolution():
    """
    Plot the average fraction of infected individuals over time for
    different initial infection counts and reproductive ratios (R0).

    Runs multiple SIS simulations across several BA networks and averages
    the infected ratio at each time step.
    """
    reproductive_ratios = [(0.1, 0.002), (0.01, 0.5)]  # (beta, mu) pairs
    initial_infected_counts = [2, 8, 40]
    n_networks = 10
    n_simulations = 10
    max_iters = 50
    final_size = 40
    initial_size = 3
    m = 2

    plt.figure(figsize=(12, 8))

    for beta, mu in reproductive_ratios:
        for initial_infected in initial_infected_counts:
            avg_infected_ratio = np.zeros(max_iters)

            for _ in range(n_networks):
                G = BA_model(initial_size, m, final_size)
                G = normalize_out_degree(G)
                for _ in range(n_simulations):
                    nodes = initial_infection(final_size, initial_infected)
                    ratio_per_sim = np.zeros(max_iters)
                    for t in range(max_iters):
                        prob = SIS_step(G, nodes, mu, beta)
                        new_state_nodes(prob, nodes)
                        ratio_per_sim[t] = np.sum(nodes == 'i') / final_size
                    avg_infected_ratio += ratio_per_sim

            avg_infected_ratio /= n_networks * n_simulations

            r0 = beta / mu
            style = '--' if r0 > 1 else '-'
            r0_label = "R0 > 1" if r0 > 1 else "R0 < 1"
            plt.plot(range(max_iters), avg_infected_ratio,
                     label=f"Initial infected: {initial_infected}, {r0_label}",
                     linestyle=style)

    plt.xlabel('Time Steps')
    plt.ylabel('Fraction of Infected Individuals')
    plt.title('Evolution of Infected Fraction over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_infection_vs_beta():
    """
    Plot the equilibrium fraction of infected individuals (rho) as a
    function of the infection rate (beta) for several recovery rates (mu).
    """
    n_networks = 10
    n_simulations = 50
    n_nodes = 40
    beta_values = np.linspace(0, 1, 25)
    mu_values = np.linspace(0.1, 0.8, 5)
    results = np.zeros((len(mu_values), len(beta_values)))

    for i, mu in enumerate(mu_values):
        for j, beta in enumerate(beta_values):
            total_infected = []
            for _ in range(n_networks):
                G = BA_model(3, 2, n_nodes)
                G = normalize_out_degree(G)
                nodes = initial_infection(n_nodes, n_nodes)
                for _ in range(n_simulations):
                    prob = SIS_step(G, nodes, mu, beta)
                    new_state_nodes(prob, nodes)
                    total_infected.append(np.sum(nodes == "i") / n_nodes)
            results[i, j] = np.mean(total_infected)

    # Normalize each mu-row by its maximum for comparability
    max_values = np.max(results, axis=1, keepdims=True)
    normalized_results = results / max_values

    plt.figure(figsize=(12, 8))
    for mu, norm_result in zip(mu_values, normalized_results):
        plt.plot(beta_values, norm_result, label=f"mu = {mu:.2f}")
    plt.xlabel('Infection rate (beta)')
    plt.ylabel('Normalized infected fraction (rho)')
    plt.title('Infected Fraction vs Infection Rate for Different Recovery Rates')
    plt.legend()
    plt.show()


def plot_infection_vs_R0():
    """
    Plot the equilibrium fraction of infected individuals (rho) as a
    function of the reproductive ratio R0 = beta / mu for different
    initial network sizes (m0).

    Uses a constant recovery rate and varies beta across 25 values.
    """
    n_networks = 10
    n_simulations = 50
    n_nodes = 40
    beta_values = np.linspace(0, 1, 25)
    mu = 0.3
    m0_values = np.arange(2, 11, 2)
    results = np.zeros((len(m0_values), len(beta_values)))

    for i, m0 in enumerate(m0_values):
        for j, beta in enumerate(beta_values):
            total_infected = []
            for _ in range(n_networks):
                G = BA_model(m0, m0 - 1, n_nodes)
                G = normalize_out_degree(G)
                nodes = initial_infection(n_nodes, n_nodes)
                for _ in range(n_simulations):
                    prob = SIS_step(G, nodes, mu, beta)
                    new_state_nodes(prob, nodes)
                    total_infected.append(np.sum(nodes == "i") / n_nodes)
            results[i, j] = np.mean(total_infected)

    plt.figure(figsize=(12, 8))
    for m0, result in zip(m0_values, results):
        plt.plot(beta_values / mu, result, label=f"m0 = {m0}")
    plt.xlabel('Reproductive ratio (R0)')
    plt.ylabel('Infected fraction (rho)')
    plt.title('Infected Fraction vs Reproductive Ratio for Different Network Densities')
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -- Part 1: Build and visualize a Barabasi-Albert network --
    allnodes = 30
    initialnodes = 2
    m = 2

    graph = BA_model(initialnodes, m, allnodes)
    plot_network(graph)
    plot_degree_histogram(graph)

    graph = normalize_out_degree(graph)

    # -- Part 2: Run a single SIS simulation and visualize each step --
    mu = 0.4
    beta = 1
    max_iters = 10

    nodes = initial_infection(allnodes)
    plot_network_colored(graph, nodes)

    for _ in range(1, max_iters):
        prob = SIS_step(graph, nodes, mu, beta)
        new_state_nodes(prob, nodes)
        plot_network_colored(graph, nodes)

    # -- Part 3: Analysis plots across multiple simulations --
    plot_infection_evolution()
    plot_infection_vs_beta()
    plot_infection_vs_R0()
