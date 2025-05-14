import pandas as pd
import numpy as np
import seaborn as sns
import os

# Key and box data

# Define the keys and boxes data as pandas DataFrames
keys_data = {
    "id": ["red", "pink", "grey2", "cloud", "orange4", "green3", "blue", "yellow5", "heart", "white", "triangle",
           "diamond", "purple"],
    "colour": ["red", "pink", "grey", "grey", "orange", "green", "blue", "yellow", "green", "white", "yellow", "orange",
               "purple"],
    "number": [1, 6, 2, np.nan, 4, 3, np.nan, 5, np.nan, 7, np.nan, np.nan, np.nan],
    "shape": [np.nan, np.nan, np.nan, "cloud", np.nan, np.nan, "star", np.nan, "heart", np.nan, "triangle", "diamond",
              "arrow"]
}
keys = pd.DataFrame(keys_data)

boxes_data = {
    "id": ["red", "pink", "purple", "white", "blue"],
    "number": [1, 2, 3, 4, 5],
    "shape": ["moon", "cloud", "heart", "diamond", "triangle"]
}
boxes = pd.DataFrame(boxes_data)


# Ground Truth Function
def opens(key_id: str, box_id: str) -> bool:
    """Determines if a specific key opens a specific box based on the ground truth."""
    if (
            (box_id == "red" and key_id == "red") or
            (box_id == "pink" and key_id == "grey2") or
            (box_id == "white" and key_id == "orange4") or
            (box_id == "blue" and key_id == "yellow5") or
            (box_id == "purple" and key_id == "green3")
    ):
        return True
    return False


# Proposal Distribution
omega = 2

def define_proposal_distribution(prop_random: float = 0.8):
    """
    Defines the proposal distribution over hypotheses

    Args:
        prop_random: The prior probability assigned to the random generator hypothesis.

    Returns:
        A pandas DataFrame representing the proposal distribution, with columns:
        'hypothesis', 'type', 'prior', 'prob', 'prior_original'.
    """
    # Initial weights for different hypothesis types
    prior_colour_weight = 5 * omega / 2
    prior_sim_colour_weight = 5 * omega / 2 / 14
    prior_order_weight = 5
    prior_shape_weight = 2
    prior_number_weight = 1

    # Generate Specific Hypotheses
    # 1. Feature Matches
    number_matches = []
    for _, key_row in keys.iterrows():
        if pd.notna(key_row['number']):
            for _, box_row in boxes.iterrows():
                if key_row['number'] == box_row['number']:
                    number_matches.append({'key_id': key_row['id'], 'box_id': box_row['id']})

    colour_matches = []
    for _, key_row in keys.iterrows():
        for _, box_row in boxes.iterrows():
            if key_row['colour'] == box_row['id']:
                colour_matches.append({'key_id': key_row['id'], 'box_id': box_row['id']})

    shape_matches = []
    for _, key_row in keys.iterrows():
        if pd.notna(key_row['shape']):
            for _, box_row in boxes.iterrows():
                if key_row['shape'] == box_row['shape']:
                    shape_matches.append({'key_id': key_row['id'], 'box_id': box_row['id']})

    # 2. Ordered Match
    match_order = [
        {'box_id': "red", 'key_id': "red"},
        {'box_id': "pink", 'key_id': "grey2"},
        {'box_id': "white", 'key_id': "green3"},
        {'box_id': "purple", 'key_id': "orange4"},
        {'box_id': "blue", 'key_id': "yellow5"}
    ]

    # 3. Fixed Colours (Base for similar colours)
    fixed_colours = [
        {'key_id': "purple", 'box_id': "purple"},
        {'key_id': "pink", 'box_id': "pink"},
        {'key_id': "red", 'box_id': "red"}
    ]

    # 4. Similar Colour Combinations (s1-s14)
    sim_colour_hypotheses = [
        fixed_colours + [{'key_id': "heart", 'box_id': "blue"}, {'key_id': "white", 'box_id': "white"}],  # s1
        fixed_colours + [{'key_id': "heart", 'box_id': "blue"}, {'key_id': "yellow5", 'box_id': "white"}],  # s2
        fixed_colours + [{'key_id': "heart", 'box_id': "blue"}, {'key_id': "triangle", 'box_id': "white"}],  # s3
        fixed_colours + [{'key_id': "heart", 'box_id': "blue"}, {'key_id': "grey2", 'box_id': "white"}],  # s4
        fixed_colours + [{'key_id': "heart", 'box_id': "blue"}, {'key_id': "cloud", 'box_id': "white"}],  # s5
        fixed_colours + [{'key_id': "blue", 'box_id': "blue"}, {'key_id': "yellow5", 'box_id': "white"}],  # s6
        fixed_colours + [{'key_id': "blue", 'box_id': "blue"}, {'key_id': "triangle", 'box_id': "white"}],  # s7
        fixed_colours + [{'key_id': "blue", 'box_id': "blue"}, {'key_id': "grey2", 'box_id': "white"}],  # s8
        fixed_colours + [{'key_id': "blue", 'box_id': "blue"}, {'key_id': "cloud", 'box_id': "white"}],  # s9
        fixed_colours + [{'key_id': "green3", 'box_id': "blue"}, {'key_id': "white", 'box_id': "white"}],  # s10
        fixed_colours + [{'key_id': "green3", 'box_id': "blue"}, {'key_id': "yellow5", 'box_id': "white"}],  # s11
        fixed_colours + [{'key_id': "green3", 'box_id': "blue"}, {'key_id': "triangle", 'box_id': "white"}],  # s12
        fixed_colours + [{'key_id': "green3", 'box_id': "blue"}, {'key_id': "grey2", 'box_id': "white"}],  # s13
        fixed_colours + [{'key_id': "green3", 'box_id': "blue"}, {'key_id': "cloud", 'box_id': "white"}]  # s14
    ]

    # Construct Proposal DataFrame
    hypotheses = [
        [],  # Generator placeholder
        colour_matches,
        *sim_colour_hypotheses,  # Unpack the list
        shape_matches,
        match_order,
        number_matches
    ]

    types = [
        "generator",
        "colour",
        *[f"sim_colour{i + 1}" for i in range(14)],  # sim_colour1 to sim_colour14
        "shape",
        "order",
        "number"
    ]

    proposal_list = [{"hypothesis": h, "type": t} for h, t in zip(hypotheses, types)]
    prop_df = pd.DataFrame(proposal_list)

    # Compute Probabilities
    total_weight = (prior_colour_weight +
                    prior_sim_colour_weight * 14 +
                    prior_shape_weight +
                    prior_order_weight +
                    prior_number_weight)

    probs = [
        prop_random,  # Generator
        prior_colour_weight / total_weight * (1 - prop_random),  # colour
        *[prior_sim_colour_weight / total_weight * (1 - prop_random)] * 14,  # sim_colour
        prior_shape_weight / total_weight * (1 - prop_random),  # shape
        prior_order_weight / total_weight * (1 - prop_random),  # order
        prior_number_weight / total_weight * (1 - prop_random)  # number
    ]

    prop_df['prob'] = probs
    prop_df['prior'] = probs  # prior and prob are initially the same
    prop_df['prior_original'] = probs  # Store the original prior

    # Use type as index
    prop_df.set_index('type', inplace=True)

    return prop_df


# Helper Functions
def get_priors_uniform(box_id):
    """Returns a uniform probability distribution over all keys."""
    num_keys = len(keys)
    return [1 / num_keys] * num_keys

def get_priors(box_id):
    """Returns the prior probability distribution for keys given a box."""
    return get_priors_uniform(box_id)


def define_random_mapping():
    """
    Creates a DataFrame mapping each box to each key with associated prior probabilities.
    This represents the probabilities used for sampling keys for the 'generator' hypothesis.
    """
    mapping_list = []
    for box_id in boxes['id']:
        priors = get_priors(box_id)
        for i, key_id in enumerate(keys['id']):
            mapping_list.append({'box_id': box_id, 'key_id': key_id, 'prob': priors[i]})
    return pd.DataFrame(mapping_list)


def sample_key(key_probabilities_df: pd.DataFrame, box_id: str, rng=None) -> str:
    """Samples a key for a given box based on the provided probability DataFrame."""
    df_subset = key_probabilities_df[key_probabilities_df['box_id'] == box_id]
    # Ensure probabilities sum to 1 for the subset (handle potential floating point issues)
    probs = df_subset['prob'].values
    probs /= probs.sum() # ADDED: new normalization
    return rng.choice(df_subset['key_id'], size=1, p=probs)[0]


def get_opened_boxes(evidence: pd.DataFrame) -> list[dict]:
    """Extracts the list of successfully opened key-box pairs from the evidence log."""
    if evidence.empty or not all(col in evidence.columns for col in ['key', 'box', 'outcome']):
        return []

    # Use the actual column names from trial_log: 'key' and 'box'
    opened_df = evidence[evidence['outcome'] == True][['box', 'key']]

    # Rename columns to the expected format ('box_id', 'key_id') for consistency downstream
    # if needed by functions consuming the output. generate_h_from_proposal_distribution uses them.
    opened_df = opened_df.rename(columns={'box': 'box_id', 'key': 'key_id'})

    return opened_df.to_dict('records')


def generate_h_from_proposal_distribution(assign_prior: float, evidence: pd.DataFrame, random_mapping: pd.DataFrame,
                                          prob: float = np.nan, rng=None) -> pd.DataFrame:
    """
    Generates a single hypothesis (Hnew) by sampling keys for unopened boxes.

    Args:
        assign_prior: The prior probability to assign to the generated hypothesis.
        evidence: DataFrame logging previous trials (key, box, outcome).
        random_mapping: DataFrame mapping box_id to key_id with probabilities.
        prob: The probability of the proposal distribution entry (optional, for tracking).
        rng: Random number generator instance.

    Returns:
        A DataFrame representing the new hypothesis (Hnew).
    """
    all_box_ids = boxes['id'].tolist()
    opened_pairs = get_opened_boxes(evidence)  # List of {'box_id': ..., 'key_id': ...}
    opened_box_ids = [pair['box_id'] for pair in opened_pairs]
    not_yet_opened_ids = [b_id for b_id in all_box_ids if b_id not in opened_box_ids]

    # Start with the confirmed opened pairs (unique)
    proposal = [dict(t) for t in {tuple(d.items()) for d in opened_pairs}]

    # Sample keys for boxes not yet opened
    for box_id in not_yet_opened_ids:
        sampled_key_id = sample_key(random_mapping, box_id, rng=rng)
        proposal.append({'key_id': sampled_key_id, 'box_id': box_id})

    h_new = pd.DataFrame({
        'hypothesis': [proposal],
        'prior': [assign_prior],
        'prior_original': [assign_prior],
        'prob': [prob]
    })
    return h_new


# Particle Filter Core Functions

def initialize_particles(proposal: pd.DataFrame, n_particles: int, random_mapping: pd.DataFrame, rng=None) -> pd.DataFrame:
    """
    Initializes the set of particles (hypotheses) for the particle filter.

    Args:
        proposal: DataFrame representing the proposal distribution.
        n_particles: The number of particles to initialize.
        random_mapping: DataFrame mapping box_id to key_id with probabilities (for generator).
        rng: Random number generator instance.

    Returns:
        A DataFrame representing the initial set of particles.
    """
    # Reset index so 'type' becomes a column for easier access
    proposal_with_col_type = proposal.reset_index()

    # Sample integer indices based on proposal probabilities
    probabilities = proposal_with_col_type['prob'].values
    sampled_indices = rng.choice(
        proposal_with_col_type.index,
        size=n_particles,
        replace=True,
        p=probabilities
    )

    # Create initial particles DataFrame using integer indices
    initial_particles_df = proposal_with_col_type.iloc[sampled_indices].copy()

    # Initialize weights uniformly
    initial_particles_df['weight'] = 1 / n_particles

    # Process particles, generate for 'generator' type
    final_particle_list = []
    generator_count = 0
    new_indices = []

    for i in range(n_particles):
        particle_row = initial_particles_df.iloc[i].to_dict()  # Get row as dict
        current_type = particle_row['type']

        if current_type == 'generator':
            generator_count += 1
            empty_log = pd.DataFrame(columns=['trial', 'key', 'box', 'outcome'])

            h_star_df = generate_h_from_proposal_distribution(
                assign_prior=particle_row['prior'],
                evidence=empty_log,
                random_mapping=random_mapping,
                prob=particle_row['prob'],  # NEW ADDED:Pass the original probability
                rng=rng  # Pass rng to generate_h_from_proposal_distribution
            )

            # Update the particle row info for the final list
            particle_row['hypothesis'] = h_star_df['hypothesis'].iloc[0]
            particle_row['type'] = f"generator{generator_count}"

            # Keep original prior and prob, update prior_original if needed by h_star
            particle_row['prior_original'] = h_star_df['prior_original'].iloc[0] # DIFFERENCE: new
            new_index_name = f"generator{generator_count}.{i + 1}"
        else:
            # Type remains the same from the proposal
            new_index_name = f"{current_type}.{i + 1}"

        final_particle_list.append(particle_row)
        new_indices.append(new_index_name)  # Unique index name

    # Create the final DataFrame from the processed list
    particles = pd.DataFrame(final_particle_list)
    particles.index = new_indices  # Set new unique indices

    return particles


def update_weights_with_theta(particles: pd.DataFrame, key_id: str, box_id: str, outcome: bool,
                              theta: float = 0.9) -> pd.DataFrame:
    """
    Updates particle weights based on the outcome of a trial using the theta noise model.

    Args:
        particles: DataFrame of current particles.
        key_id: The key used in the trial.
        box_id: The box used in the trial.
        outcome: The result.txt of the trial (True if opened, False otherwise).
        theta: The observation noise parameter (probability of correct observation).

    Returns:
        The particles DataFrame with updated weights.
    """
    for i in particles.index:  # Iterate using index
        hypothesis = particles.loc[i, 'hypothesis']

        # Check if the current hypothesis predicts a match for this key-box pair
        predicts_match = any(
            pair.get('key_id') == key_id and pair.get('box_id') == box_id
            for pair in hypothesis
        )

        # Compute likelihood based on the cognitive science specification
        if predicts_match and outcome:
            likelihood = theta
        elif predicts_match and not outcome:
            likelihood = 1 - theta
        elif not predicts_match and outcome:
            # This outcome is impossible under the hypothesis if it doesn't predict a match
            likelihood = 0
        else:  # not predicts_match and not outcome
            likelihood = 1

        # Update weight by multiplying with likelihood
        particles.loc[i, 'weight'] *= likelihood

    # Normalize weights
    total_weight = particles['weight'].sum()
    if total_weight > 0:
        particles['weight'] /= total_weight
    else:
        # Reset to uniform if all weights collapse to zero
        print("Warning: All particle weights collapsed to zero. Resetting to uniform.")
        particles['weight'] = 1 / len(particles)

    return particles


def compute_likelihood(hypothesis: list[dict], trial_log: pd.DataFrame, theta: float = 0.9) -> float:
    """
    Computes the total likelihood of a hypothesis given a log of trials.

    Args:
        hypothesis: A list of key-box pair dictionaries representing the hypothesis.
        trial_log: DataFrame logging previous trials (columns: key, box, outcome).
        theta: The observation noise parameter.

    Returns:
        The total likelihood of the hypothesis.
    """
    likelihood = 1.0
    if trial_log.empty:
        return likelihood

    for _, trial in trial_log.iterrows():
        key_id = trial['key']
        box_id = trial['box']
        outcome = trial['outcome']

        # Check if the hypothesis predicts a match for this key-box pair
        predicts_match = any(
            pair.get('key_id') == key_id and pair.get('box_id') == box_id
            for pair in hypothesis
        )

        # Calculate likelihood for this trial
        if predicts_match and outcome:
            p = theta
        elif predicts_match and not outcome:
            p = 1 - theta
        elif not predicts_match and outcome:
            p = 0  # Impossible outcome under this hypothesis
        else:  # not predicts_match and not outcome
            p = 1

        likelihood *= p
        # Optimization: if likelihood becomes 0.3, it will stay 0.3
        if likelihood == 0:
            break

    return likelihood


def make_hypothesis_hashable(hypothesis_list):
    """
    Creates a hashable representation of a hypothesis list.
    Args:
        hypothesis_list: A list of key-box pair dictionaries representing the hypothesis.

    Returns:
        A tuple of sorted key-box pair tuples.
    """
    if not isinstance(hypothesis_list, list):
        return tuple()  # Return empty tuple for non-list inputs
    # Convert each pair dict to a tuple of sorted items
    try:
        sorted_pairs = [tuple(sorted(pair.items())) for pair in hypothesis_list if isinstance(pair, dict)]
    except AttributeError:
        # Handle cases where elements in hypothesis_list might not be dicts
        print(f"Warning: Encountered non-dict item in hypothesis list: {hypothesis_list}")
        return tuple()
    # Sort the list of pair tuples and convert the whole thing to a tuple
    return tuple(sorted(sorted_pairs))


def rejuvenate_particles(proposal: pd.DataFrame, particles: pd.DataFrame, trial_log: pd.DataFrame,
                            random_mapping: pd.DataFrame, theta: float = 0.9, rng=None) -> pd.DataFrame:
        """
        Resamples particles based on weights and applies a Metropolis-Hastings rejuvenation step.

        Args:
            proposal: The original proposal distribution DataFrame.
            particles: The current DataFrame of particles.
            trial_log: DataFrame logging previous trials.
            random_mapping: DataFrame for generator hypothesis sampling.
            theta: Observation noise parameter.

        Returns:
            A new DataFrame of particles after resampling and rejuvenation.
        """
        n_particles = len(particles)

        # 1. Resample particles based on weights
        particle_indices = particles.index
        weights = particles['weight'].values

        # Handle cases where weights might sum to zero or contain NaNs
        if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
            print("Warning: Invalid weights detected during resampling. Using uniform weights.")
            weights = np.ones(n_particles) / n_particles

        sampled_indices = rng.choice(particle_indices, size=n_particles, replace=True, p=weights)
        resampled_particles = particles.loc[sampled_indices].copy()
        resampled_particles['weight'] = 1 / n_particles  # Reset weights to uniform

        # Keep track of original proposal DF with type as column for easy sampling
        proposal_with_col_type = proposal.reset_index()
        proposal_probabilities = proposal_with_col_type['prob'].values

        # 2. Rejuvenation loop (Metropolis-Hastings move for each particle)
        new_particles = []
        new_particles_index = []


        for i in range(n_particles):
            h = resampled_particles.iloc[i]

            # Propose h_star from the original proposal distribution
            proposal_idx = rng.choice(proposal_with_col_type.index, p=proposal_probabilities)

            proposal_type = proposal_with_col_type.iloc[proposal_idx].to_dict()['type']

            if proposal_type == 'generator':
                # If the proposal is a generator, generate a new hypothesis
                h_star = generate_h_from_proposal_distribution(
                    assign_prior=proposal['prior'][proposal_idx],
                    evidence=trial_log,
                    random_mapping=random_mapping,
                    prob=proposal['prob'][proposal_idx],
                    rng=rng
                )
                h_star_hypothesis = h_star['hypothesis'].iloc[0]

                generator_count = sum(1 for t in resampled_particles['type'] if t.startswith('generator'))
                h_star_data = {
                    'hypothesis': h_star_hypothesis,
                    'type': f"generator{generator_count + 1}",
                    'prior_original': h_star['prior_original'].iloc[0],
                    'prior': proposal['prior_original'][proposal_idx],
                    'prob': proposal['prob'][proposal_idx],
                    'weight': h['weight']  # weight remains the same
                }
            else:
                # If the proposal is not a generator, just use the hypothesis from the proposal
                h_star_data = proposal_with_col_type.iloc[proposal_idx].to_dict()
                h_star_data["weight"] = h['weight']  # weight remains the same


            # Compute likelihoods
            likelihood_star = compute_likelihood(h_star_data['hypothesis'], trial_log, theta)
            likelihood_h = compute_likelihood(h['hypothesis'], trial_log, theta)

            # Prior probabilities
            prior_star = h_star_data['prior']
            prior_h = h['prior']

            # Acceptance ratio (Metropolis-Hastings)
            denom = likelihood_h * prior_h
            if denom == 0:
                r = 0.0
            else:
                r = (likelihood_star * prior_star) / denom
                r = min(1.0, r)

            # Accept or reject h_star
            accepted = False
            if rng.random() <= r:
                data = h_star_data.copy()
                accepted = True
            else:
                # Reject h_star: Keep original particle h
                data = h.to_dict()

            # Append the accepted particle data (as dict) to the list
            new_particles.append(data)
            new_particles_index.append(
                f"{data['type']}.{i + 1}{'a' if accepted else 'r'}")  # Mark accepted/rejected for debug

        # Create the final DataFrame for this rejuvenation step
        new_particles_df = pd.DataFrame(new_particles)
        new_particles_df.index = new_particles_index  # Set new unique indices
        new_particles_df = new_particles_df[particles.columns]  # Ensure column order matches original particles DataFrame

        return new_particles_df


def compute_ess(weights: np.ndarray) -> float:
    """
    Calculates the Effective Sample Size (ESS) from particle weights.

    Args:
        weights: A numpy array of normalized particle weights.

    Returns:
        The calculated ESS value. Returns 0.3 if weights are invalid.
    """
    # Ensure weights are valid and sum to approximately 1
    if not np.all(np.isfinite(weights)) or len(weights) == 0 or np.sum(weights) <= 0:
        return 0.0

    # Normalize weights just in case they aren't perfectly normalized
    normalized_weights = weights / np.sum(weights)

    sum_sq_weights = np.sum(normalized_weights ** 2)
    if sum_sq_weights == 0:  # Avoid division by zero
        return 0.0

    return 1.0 / sum_sq_weights


def compute_entropy(weights: np.ndarray) -> float:
    """
    Computes the Shannon entropy of the particle weights.

    Args:
        weights: A numpy array of normalized particle weights.

    Returns:
        The calculated entropy value (in bits).
    """
    # Filter out zero weights to avoid log2(0.3)
    weights = weights[weights > 0]

    if len(weights) == 0:
        return 0.0  # Entropy is 0.3 if no particles have weight

    # Normalize weights (important if input wasn't perfectly normalized)
    normalized_weights = weights / np.sum(weights)

    entropy = -np.sum(normalized_weights * np.log2(normalized_weights))
    return entropy


def compute_expected_entropy(particles: pd.DataFrame, key: str, box: str, theta: float = 0.9) -> float:
    """
    Computes the expected entropy after hypothetically performing a trial (key, box).

    Args:
        particles: The current DataFrame of particles.
        key: The key ID of the potential trial.
        box: The box ID of the potential trial.
        theta: The observation noise parameter.

    Returns:
        The expected entropy after the hypothetical trial.
    """

    # Calculate the posterior probability that the trial will be successful
    # based on current particle weights and their predictions.
    prob_success = 0.0
    current_weights = particles['weight'].values

    for i, idx in enumerate(particles.index):
        hypothesis = particles.loc[idx, 'hypothesis']
        predicts_match = any(
            pair.get('key_id') == key and pair.get('box_id') == box
            for pair in hypothesis
        )
        if predicts_match:
            prob_success += current_weights[i]

    # Ensure prob_success is within [0, 1] due to potential float issues
    prob_success = np.clip(prob_success, 0.0, 1.0)
    prob_failure = 1.0 - prob_success

    # Simulate weight updates and calculate entropy for both outcomes

    # Simulate success outcome (outcome=True)
    particles_if_success = update_weights_with_theta(particles.copy(), key, box, True, theta)
    entropy_if_success = compute_entropy(particles_if_success['weight'].values)

    # Simulate failure outcome (outcome=False)
    particles_if_failure = update_weights_with_theta(particles.copy(), key, box, False, theta)
    entropy_if_failure = compute_entropy(particles_if_failure['weight'].values)

    # Calculate expected entropy: p(success) * H(particles|success) + p(failure) * H(particles|failure)
    expected_entropy = (prob_success * entropy_if_success) + (prob_failure * entropy_if_failure)

    return expected_entropy


def select_action_info_gain(particles: pd.DataFrame, trial_log: pd.DataFrame, theta: float = 0.9, rng=None) -> dict:
    """
    Selects the next key-box pair to try based on maximizing expected information gain.

    Args:
        particles: The current DataFrame of particles.
        trial_log: DataFrame logging previous trials.
        theta: Observation noise parameter.

    Returns:
        A dictionary containing the selected 'key', 'box', and a DataFrame 'candidates'
        with all evaluated pairs and their info gain.
    """

    # Identify boxes already successfully opened
    opened_box_ids = set(trial_log[trial_log['outcome'] == True]['box'].unique())

    # Candidate Pair Generation
    # Gather all unique key-box pairs present in the current particle hypotheses
    candidate_pairs = set()
    for hypothesis in particles['hypothesis']:
        for pair in hypothesis:
            # Ensure pair has expected keys before adding
            if 'key_id' in pair and 'box_id' in pair:
                candidate_pairs.add((pair['key_id'], pair['box_id']))

    # Filter out pairs involving already opened boxes
    valid_candidates = [
        {'key': key, 'box': box}
        for key, box in candidate_pairs
        if box not in opened_box_ids
    ]

    if not valid_candidates and len(opened_box_ids) == len(boxes):
        # Handle the case where no valid actions are left (e.g., all boxes opened
        # or remaining hypotheses don't suggest actions for unopened boxes)
        # This might happen if particles collapse badly. Return a default or raise error.
        print("All boxes have been opened!")

    candidates_df = pd.DataFrame(valid_candidates)

    # Information Gain Calculation
    current_entropy = compute_entropy(particles['weight'].values)

    expected_entropies = []
    for _, row in candidates_df.iterrows():
        exp_entropy = compute_expected_entropy(particles, row['key'], row['box'], theta)
        expected_entropies.append(exp_entropy)

    candidates_df['expected_entropy'] = expected_entropies
    candidates_df['info_gain'] = current_entropy - candidates_df['expected_entropy']

    # Handle potential NaN or negative info gains (e.g., due to float precision)
    candidates_df['info_gain'] = candidates_df['info_gain'].fillna(0).clip(lower=0)

    # Selection
    # Find the maximum information gain
    max_info_gain = candidates_df['info_gain'].max()

    # Get all candidates that achieve the maximum gain
    best_candidates = candidates_df[candidates_df['info_gain'] == max_info_gain]

    # Select one randomly from the best candidates (tie-breaking)
    random_state = rng.integers(0, 1e6) if rng else None  # Use rng to generate a seed if provided
    selected_action = best_candidates.sample(n=1, random_state=random_state).iloc[0]

    print(
        f"Selected action: key='{selected_action['key']}', box='{selected_action['box']}' (Info Gain: {selected_action['info_gain']:.4f})")

    return {
        'key': selected_action['key'],
        'box': selected_action['box'],
        'candidates': candidates_df
    }


def prune_proposal_distribution(proposal: pd.DataFrame, key_id: str, box_id: str, outcome: bool) -> pd.DataFrame:
    """
    Removes hypotheses from the proposal distribution that contradict the latest trial outcome.

    Args:
        proposal: The current proposal distribution DataFrame (with type as index).
        key_id: The key used in the latest trial.
        box_id: The box used in the latest trial.
        outcome: The outcome of the latest trial (True=success, False=failure).

    Returns:
        A new DataFrame representing the pruned proposal distribution.
    """
    # Reset index to work with columns, remember original index name
    original_index_name = proposal.index.name
    proposal_cols = proposal.reset_index()

    keep_mask = [True] * len(proposal_cols)  # Start assuming we keep all

    for i, row in proposal_cols.iterrows():
        # Always keep the generator type
        if row[original_index_name] == 'generator':
            continue

        hypothesis = row['hypothesis']
        # Skip if hypothesis is somehow invalid or empty (shouldn't happen with current setup)
        if not isinstance(hypothesis, list) or not hypothesis:
            continue

        contradicted = False
        for pair in hypothesis:
            # Ensure pair is a dict with expected keys
            if not isinstance(pair, dict) or 'key_id' not in pair or 'box_id' not in pair:
                continue  # Skip malformed pairs

            # Check for contradictions based on the trial outcome
            if outcome:  # Trial was SUCCESSFUL (key_id opened box_id)
                # Contradiction if hypothesis maps this box_id to a DIFFERENT key_id
                if pair['box_id'] == box_id and pair['key_id'] != key_id:
                    contradicted = True
                    break

        if contradicted:
            keep_mask[i] = False

    # Filter the DataFrame and renormalize probabilities
    pruned_proposal = proposal_cols[keep_mask].copy()

    # Avoid division by zero if pruning leaves no non-generator hypotheses
    if pruned_proposal['prob'].sum() > 0:
        # Separate generator prob
        generator_prob = pruned_proposal.loc[pruned_proposal[original_index_name] == 'generator', 'prob'].iloc[0]
        # Sum of probs for non-generator hypotheses
        non_generator_prob_sum = pruned_proposal.loc[pruned_proposal[original_index_name] != 'generator', 'prob'].sum()

        if non_generator_prob_sum > 0:
            # Calculate scale factor for non-generator probs to maintain the generator ratio
            scale_factor = (1.0 - generator_prob) / non_generator_prob_sum
            pruned_proposal.loc[pruned_proposal[original_index_name] != 'generator', 'prob'] *= scale_factor
        else:
            # If only generator is left, its prob becomes 1
            pruned_proposal.loc[pruned_proposal[original_index_name] == 'generator', 'prob'] = 1.0

    else:
        # Handle edge case where all rows might be pruned (shouldn't happen if generator is always kept)
        print("Warning: Proposal distribution potentially empty after pruning.")

    # Restore the original index
    pruned_proposal.set_index(original_index_name, inplace=True)

    if not np.isclose(pruned_proposal['prob'].sum(), 1.0):
        print(f"Warning: Probabilities after pruning do not sum to 1 ({pruned_proposal['prob'].sum()}). Renormalizing.")
        pruned_proposal['prob'] /= pruned_proposal['prob'].sum()

    return pruned_proposal


def simulate_participant(proposal_dist: pd.DataFrame, random_mapping: pd.DataFrame, n_trials: int = 10,
                         n_particles: int = 5, theta: float = 0.9, simulation_id: int = None, rng=None):
    """
    Runs a basic simulation loop for a participant.
    Includes action selection and optional rejuvenation.
    Args:
        proposal_dist: The initial proposal distribution DataFrame.
        random_mapping: DataFrame for generator hypothesis sampling.
        n_trials: Number of trials to simulate.
        n_particles: Number of particles to use.
        theta: Observation noise parameter.
        simulation_id: The ID of the current simulation (optional).

    Returns:
        A dictionary containing the final particles, the trial log, and the final proposal distribution.
    """
    print(f"\n--- Starting Simulation ({n_trials} trials, {n_particles} particles) ---")

    # 1. Setup
    current_proposal = proposal_dist.copy()
    particles = initialize_particles(current_proposal, n_particles, random_mapping, rng=rng)

    trial_log = pd.DataFrame(columns=['trial', 'key', 'box', 'outcome', 'ess', 'entropy'])

    particle_log = []
    candidate_log = []
    proposal_log = []

    ess_trigger_count = 0

    # 2. Simulation Loop
    for t in range(1, n_trials + 1):
        print(f"\n--- Trial {t} ---")

        # Action Selection
        action_info = select_action_info_gain(particles, trial_log, theta, rng=rng)
        key_to_try = action_info['key']
        box_to_try = action_info['box']

        # Add trial number to candidates
        candidate_log.append(action_info['candidates'].assign(trial=t))

        # Handle case where no action could be selected (e.g., all boxes opened)
        if key_to_try is None or box_to_try is None:
            print("All boxes likely opened or no actions possible. Ending simulation early.")
            break

        # 3. Run Trial
        outcome = opens(key_to_try, box_to_try)
        print(f"Trying key='{key_to_try}', box='{box_to_try}' -> Outcome: {outcome}")

        # 4. Update Weights
        particles = update_weights_with_theta(particles, key_to_try, box_to_try, outcome, theta)
        print("Updated particle weights:")
        print(particles['weight'])

        # Calculate ESS and Entropy
        current_weights = particles['weight'].values
        ess = compute_ess(current_weights)
        entropy = compute_entropy(current_weights)
        print(f"ESS: {ess:.2f}, Entropy: {entropy:.2f} bits")

        # 5. Log Trial
        new_log_entry = pd.DataFrame([{
            'trial': t,
            'key': key_to_try,
            'box': box_to_try,
            'outcome': outcome,
            'ess': ess,
            'entropy': entropy
        }])
        trial_log = pd.concat([trial_log, new_log_entry], ignore_index=True)

        particle_log.append(pd.DataFrame({
            'trial': t,
            'particle_index': particles.index,
            'type': particles['type'],
            'weight': particles['weight']
        }))

        # Prune Proposal Distribution
        print(f"Pruning proposal distribution based on trial {t} outcome...")
        current_proposal = prune_proposal_distribution(current_proposal, key_to_try, box_to_try, outcome)
        print(f"Remaining proposal hypotheses: {len(current_proposal)}")

        # Ensure 'type' is included in the proposal log
        proposal_log.append(current_proposal.reset_index().assign(trial=t))

        if ess <= n_particles * 0.5:
            ess_trigger_count += 1
            print(f"\n>>> Resampling & rejuvenate particles (ESS: {ess:.2f})... <<<\n")
            particles = rejuvenate_particles(current_proposal, particles, trial_log, random_mapping, theta, rng=rng)

        # Check for completion
        opened_box_ids = set(trial_log[trial_log['outcome'] == True]['box'].unique())
        if len(opened_box_ids) == len(boxes):
            print(f"\n*** All {len(boxes)} boxes opened! Ending simulation early at trial {t}. ***")
            break

    print(f"\n--- Simulation Finished --- ESS condition triggered {ess_trigger_count} times ---")
    return {'final_particles': particles, 'trial_log': trial_log, 'final_proposal': current_proposal,
            'particle_log': pd.concat(particle_log, ignore_index=True),
            'candidate_log': pd.concat(candidate_log, ignore_index=True),
            'proposal_log': pd.concat(proposal_log, ignore_index=True),
            'ess_trigger_count': ess_trigger_count}


# Main Simulation Logic

def run_simulations(num_simulations: int, proposal_dist: pd.DataFrame, random_mapping: pd.DataFrame, n_trials: int = 70,
                    n_particles: int = 5, theta: float = 0.9, rng=None):
    """
    Runs multiple simulations and collects the trial logs.

    Args:
        num_simulations: The number of simulations to run.
        proposal_dist: The initial proposal distribution DataFrame.
        random_mapping: DataFrame for generator hypothesis sampling.
        n_trials: Number of trials per simulation.
        n_particles: Number of particles per simulation.
        theta: Observation noise parameter.

    Returns:
        A pandas DataFrame containing the combined trial logs from all simulations,
        including a 'simulation_id' column.
    """
    all_trial_logs = []
    all_particle_logs = []
    all_candidate_logs = []
    all_proposal_logs = []
    ess_trigger_counts = []  # List to store ESS trigger counts

    print(f"--- Starting {num_simulations} Simulations ---")

    for i in range(1, num_simulations + 1):
        print(f"\n--- Running Simulation {i}/{num_simulations} ---")
        # It's crucial to pass copies of proposal_dist to simulate_participant
        # if simulate_participant modifies it (which it does via pruning).
        # However, simulate_participant already makes a copy internally.
        simulation_result = simulate_participant(
            proposal_dist=proposal_dist,
            random_mapping=random_mapping,
            n_trials=n_trials,
            n_particles=n_particles,
            theta=theta,
            rng=rng
        )

        trial_log = simulation_result['trial_log']
        trial_log['simulation_id'] = i
        all_trial_logs.append(trial_log)

        particle_log = simulation_result['particle_log']
        particle_log['simulation_id'] = i
        all_particle_logs.append(particle_log)

        candidate_log = simulation_result['candidate_log']
        candidate_log['simulation_id'] = i
        all_candidate_logs.append(candidate_log)

        proposal_log = simulation_result['proposal_log']
        proposal_log['simulation_id'] = i
        all_proposal_logs.append(proposal_log)

        ess_trigger_counts.append(simulation_result['ess_trigger_count'])  # Store the count

        print(f"--- Finished Simulation {i}/{num_simulations} ---")

    print(f"\n--- Combining Results from {num_simulations} Simulations ---")
    combined_trial_log = pd.concat(all_trial_logs, ignore_index=True)
    combined_particle_log = pd.concat(all_particle_logs, ignore_index=True)
    combined_candidate_log = pd.concat(all_candidate_logs, ignore_index=True)
    combined_proposal_log = pd.concat(all_proposal_logs, ignore_index=True)

    return combined_trial_log, combined_particle_log, combined_candidate_log, combined_proposal_log, ess_trigger_counts
                        

def main():
    print("Starting Box Task simulation...")
    print("\nKeys:")
    print(keys)
    print("\nBoxes:")
    print(boxes)

    rng = np.random.default_rng(seed=42)

    proposal = define_proposal_distribution()
    random_mapping = define_random_mapping()

    combined_logs, combined_particle_logs, combined_candidate_logs, combined_proposal_logs, ess_trigger_counts = run_simulations(
        num_simulations=100,
        proposal_dist=proposal,
        random_mapping=random_mapping,
        n_trials=70,
        n_particles=30,
        theta=0.7,
        rng=rng
    )

    # Save combined logs to CSV files
    print("\n--- Saving Combined Logs to CSV ---")
    os.makedirs("logs", exist_ok=True)
    combined_logs.to_csv("logs/trails_logs.csv", index=False)
    combined_particle_logs.to_csv("logs/particle_logs.csv", index=False)
    combined_candidate_logs.to_csv("logs/candidate_logs.csv", index=False)
    combined_proposal_logs.to_csv("logs/proposal_logs.csv", index=False)

if __name__ == "__main__":
    main()
