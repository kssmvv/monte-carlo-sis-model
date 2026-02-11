# SIS Epidemic Simulation on Scale-Free Networks

Monte Carlo simulation of disease spread on Barabási–Albert scale-free networks using the **SIS (Susceptible-Infected-Susceptible)** epidemic model.

## Overview

This project models how infectious diseases propagate through a population by combining two ideas:

1. **Barabási–Albert (BA) networks** — Real-world contact networks are not random; a few individuals have many connections while most have few. The BA model captures this by growing a network through preferential attachment, producing the characteristic power-law degree distribution seen in social and biological networks.

2. **SIS epidemic dynamics** — Each individual (node) is either **Susceptible** or **Infected**. Infected nodes can transmit the disease to susceptible neighbors with probability proportional to the infection rate β, and recover back to susceptible with recovery rate μ. Unlike the SIR model, there is no permanent immunity — recovered individuals can be reinfected.

The simulation runs Monte Carlo trials across multiple randomly generated networks and averages the results to study how the basic reproductive ratio R0 = β/μ, the infection rate, and the network structure influence the long-term fraction of infected individuals.

## What the Code Produces

The script generates six types of plots:

| Plot | Description |
|------|-------------|
| **Network graph** | Visualization of a BA network with 30 nodes |
| **Degree distribution** | Histogram confirming the scale-free (power-law) property |
| **Infection snapshots** | Network drawn at each SIS time step — infected nodes in red, susceptible in blue |
| **Infection evolution** | Average infected fraction over 50 time steps for different initial conditions and R0 regimes (R0 < 1 vs R0 > 1) |
| **ρ vs β** | Equilibrium infected fraction as a function of the infection rate for several recovery rates |
| **ρ vs R0** | Equilibrium infected fraction as a function of R0 for networks with different initial densities (m0) |

## Requirements

- Python 3.8+
- numpy
- networkx
- matplotlib
- colorama

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python monte_carlo_simulation.py
```

The script will generate and display all plots sequentially. The analysis plots (infection evolution, ρ vs β, ρ vs R0) run many simulations and may take a few minutes to complete.

## Project Structure

```
├── monte_carlo_simulation.py   # Main simulation script
├── requirements.txt            # Python dependencies
└── README.md
```

