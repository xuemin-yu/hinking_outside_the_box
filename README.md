# thinking_outside_the_box

This code implements a cognitive science experiment called the "Box Task" using Sequential Monte Carlo (SMC) with Metropolis-Hastings (MH) sampling to model how people learn which keys open which boxes.

## Goal
To simulate how people might learn the relationship between keys and boxes under uncertainty using a Bayesian cognitive model.

## Setup
- 13 keys with different properties (color, number, shape)
- 5 boxes with different properties 
- A fixed ground truth about which key opens which box (defined in the `open()` function)

## Methods
1. **Bayesian particle filtering**:
   - Maintains a set of weighted hypotheses (particles) about key-box relationships
   - Updates belief weights based on evidence using Bayes' rule

2. **Information gain** to select actions:
   - Chooses key-box pairs that maximize expected information gain
   - Computes entropy before and after potential actions

3. **Particle rejuvenation** via Metropolis-Hastings:
   - Proposes new hypotheses and accepts/rejects based on likelihood ratio
   - Prevents particle degeneracy (when most weight concentrates on few particles)

## Flow
1. Initialize particle hypotheses from a proposal distribution
2. For each trial:
   - Select key-box pair with highest information gain
   - Test if key opens box according to ground truth
   - Update particle weights based on outcome
   - Occasionally rejuvenate particles using MH sampling
   - Stop when all boxes opened or max trials reached
3. Run multiple simulations and analyze:
   - Trials needed to open all boxes
   - Repeated attempts
   - Most likely hypothesis types
   - Learning efficiency

The final part analyzes simulation results to understand model behavior.
