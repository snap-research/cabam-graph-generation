import logging

import networkx as nx
import numpy as np
from tqdm import tqdm


class CABAM(object):
    def generate_graph(self, n, m, num_classes=2, native_class_probs=[0.5, 0.5], inter_intra_link_probs={1: 0.5, 0: 0.5}):
        """
        Main function for CABAM graph generation.

        n: maximum number of nodes
        m: number of edges to add at each timestep (also the minimum degree)
        num_classes: number of classes
        native_class_probs: c-length vector of native class probabilities (must sum to 1)
        inter_intra_link_probs: p_c from the paper.  Entry for 1 (0) is the intra-class (inter-class) link strength.  Entries must sum to 1.

        Supports 3 variants of c_probs:
        -Callable (degree-dependent). Ex: c_probs = lambda k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)}
        -Precomputed (degree-dependent) dictionary.  Ex: c_probs = {k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)} for k in range(100)}
        -Fixed (constant).  Ex: c_probs = {1: p_c, 0: 1 - p_c}
        """

        if m < 1 or n < m:
            raise nx.NetworkXError(
                f"Must have m>1 and m<n, m={m},n={n}"
            )

        # Initialize graph.
        G = nx.empty_graph(m)
        intra_class_edges = 0
        inter_class_edges = 0
        total_intra_class_edges = 0
        total_inter_class_edges = 0

        class_tbl = list(range(num_classes))
        node_labels = np.array(
            [
                np.random.choice(class_tbl, p=native_class_probs)
                for _ in range(G.number_of_nodes())
            ]
        )
        node_degrees = np.array(
            [1] * m
        )  # technically degree 0, but using 1 here to make the math work out.

        # start adding nodes
        source = m
        source_label = np.random.choice(class_tbl, p=native_class_probs)
        pbar = tqdm(total=n)
        pbar.update(m)
        empirical_edge_fraction_to_degree_k = np.zeros(10)
        n_added = 0

        def _resolve_node_class_probs(labels, degrees, src_label):
            if isinstance(inter_intra_link_probs, dict):
                if len(inter_intra_link_probs) == 2:
                    # Handling constant inter-intra link probs.
                    _node_class_probs = np.array(
                        [
                            inter_intra_link_probs[abs(labels[i] == src_label)]
                            for i in range(len(labels))
                        ]
                    )
                else:
                    # Handling pre-generated, custom inter-intra link probs.
                    _node_class_probs = np.array(
                        [
                            inter_intra_link_probs[degrees[i]][abs(labels[i] == src_label)]
                            for i in range(len(labels))
                        ]
                    )
            else:
                # Handling callable, dynamic inter-intra link probs.
                _node_class_probs = np.array(
                    [
                        inter_intra_link_probs(degrees[i])[abs(labels[i] == src_label)]
                        for i in range(len(labels))
                    ]
                )
            return _node_class_probs

        while source < n:
            logging.debug(f"Adding node {source} with label {source_label}.")
            node_class_probs = _resolve_node_class_probs(
                labels=node_labels, degrees=node_degrees, src_label=source_label
            )

            # Determine m target nodes for source node to connect to.
            targets = []
            while len(targets) != m:
                node_class_degree_probs = node_class_probs * node_degrees
                candidate_targets = np.where(node_class_degree_probs > 0)[0]

                if len(candidate_targets) >= m:
                    logging.debug("Have enough targets...sampling from assortativity-weighted PA probs.\n")
                    # If we have enough qualifying nodes, sample from assortativity-weighted PA probs
                    candidate_node_class_degree_probs = node_class_degree_probs[
                        candidate_targets
                    ]
                    candidate_node_class_degree_probs = (
                            candidate_node_class_degree_probs
                            / np.linalg.norm(node_class_degree_probs, ord=1)
                    )
                    targets = np.random.choice(
                        candidate_targets,
                        size=m,
                        p=candidate_node_class_degree_probs,
                        replace=False,
                    )
                else:
                    logging.debug("Not enough targets...falling back to PA probs.\n")
                    # Else, use as many qualifying nodes as possible, and just sample from the PA probs for the rest.
                    n_remaining_targets = m - len(candidate_targets)
                    other_choices = np.where(node_class_degree_probs == 0)[0]
                    other_node_degree_probs = node_degrees[other_choices]
                    other_node_degree_probs = other_node_degree_probs / np.linalg.norm(
                        other_node_degree_probs, ord=1
                    )
                    other_targets = np.random.choice(
                        other_choices,
                        size=n_remaining_targets,
                        p=other_node_degree_probs,
                        replace=False,
                    )
                    targets = np.concatenate((candidate_targets, other_targets))
                assert len(targets) == m

            G.add_edges_from([(source, target) for target in targets])
            is_intra_class_edge = np.array(
                [source_label == node_labels[target] for target in targets]
            )
            num_intra_class_edges_from_targets = np.count_nonzero(is_intra_class_edge)
            num_inter_class_edges_from_targets = np.count_nonzero(is_intra_class_edge == 0)
            intra_class_edges += num_intra_class_edges_from_targets
            inter_class_edges += num_inter_class_edges_from_targets

            total_intra_class_edges += num_intra_class_edges_from_targets
            total_inter_class_edges += num_inter_class_edges_from_targets
            total_intra_frac = total_intra_class_edges / (total_intra_class_edges + total_inter_class_edges)

            normed_node_class_degree_probs = node_class_degree_probs / np.linalg.norm(node_class_degree_probs, ord=1)
            empirical_edge_fraction_to_degree_k += np.array(
                [m * np.sum(normed_node_class_degree_probs[node_degrees == k]) for k in range(m, m + 10)]
            )

            if source % 500 == 0:
                theoretical_edge_fraction_to_degree_k = [
                    ((m * (m + 1)) / ((k + 1) * (k + 2))) for k in range(m, m + 10)
                ]
                avgd_empirical_edge_fraction_to_degree_k = (
                        empirical_edge_fraction_to_degree_k / n_added
                )

                logging.debug(
                    f"Theor. edge prob to deg k: {np.round(theoretical_edge_fraction_to_degree_k, 3)}"
                )
                logging.debug(
                    f"Empir. edge prob to deg k: {np.round(avgd_empirical_edge_fraction_to_degree_k, 3)}"
                )
                snapshot_intra_frac = intra_class_edges / (
                        intra_class_edges + inter_class_edges
                )
                logging.debug(
                    f"Snapshot: ({intra_class_edges}/{intra_class_edges + inter_class_edges})"
                    f"={snapshot_intra_frac:.3f}\t Overall: {total_intra_frac:.3f}"
                )
                intra_class_edges = 0
                inter_class_edges = 0
                logging.debug(f"Max node degree: {max(node_degrees)}")

            # Bookkeeping.
            node_degrees[targets] += 1
            node_labels = np.append(node_labels, source_label)
            node_degrees = np.append(node_degrees, m)
            pbar.update(1)

            # Move onto next node.
            n_added += 1
            source += 1
            source_label = np.random.choice(class_tbl, p=native_class_probs)

        pbar.close()
        return (
            G,
            node_degrees,
            node_labels,
            total_intra_class_edges,
            total_inter_class_edges,
        )

    @staticmethod
    def get_theoretical_intra_class_estimates(inter_intra_link_probs, native_class_probs, m, approx_limit=100000):
        """
        Estimate the theoretical (in-the-limit) intra-class edge probabilities.
        Has some variance due to the stochasticity in empirical power-law generation
        and theoretical probabilities from the underlying BA model.

        Note: Supports 3 variants of c_probs:
        -Callable (degree-dependent). Ex: c_probs = lambda k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)}
        -Precomputed (degree-dependent) dict.  Ex: c_probs = {k: {1: np.tanh(k/5), 0: 1 - np.tanh(k/5)} for k in range(100)}
        -Fixed (constant).  Ex: c_probs = {1: p_c, 0: 1 - p_c}

        `approx_limit` is used to approximate the infinite sums in Eqs. 5-6 from the paper.
        """

        g = np.sum(
            np.power(native_class_probs, 2)
        )  # probability of an intra-class edge to arise given certain class propensities
        if callable(inter_intra_link_probs):
            # Dynamically computed, degree dependent
            within = np.sum(
                [
                    (g * inter_intra_link_probs(k)[1] * (m + 1)) / ((k + 1) * (k + 2))
                    for k in range(m, approx_limit)
                ]
            )
            cross = np.sum(
                [
                    ((1 - g) * (1 - inter_intra_link_probs(k)[1]) * (m + 1)) / ((k + 1) * (k + 2))
                    for k in range(m, approx_limit)
                ]
            )
        else:
            if len(inter_intra_link_probs) == 2:
                # Fixed
                within = np.sum(
                    [
                        (g * inter_intra_link_probs[1] * (m + 1)) / ((k + 1) * (k + 2))
                        for k in range(m, approx_limit)
                    ]
                )
                cross = np.sum(
                    [
                        ((1 - g) * (1 - inter_intra_link_probs[1]) * (m + 1)) / ((k + 1) * (k + 2))
                        for k in range(m, approx_limit)
                    ]
                )
            else:
                # Precomputed, degree-dependent
                within = np.sum(
                    [
                        (g * inter_intra_link_probs[k][1] * (m + 1)) / ((k + 1) * (k + 2))
                        for k in range(m, approx_limit)
                    ]
                )
                cross = np.sum(
                    [
                        ((1 - g) * (1 - inter_intra_link_probs[k][1]) * (m + 1)) / ((k + 1) * (k + 2))
                        for k in range(m, approx_limit)
                    ]
                )

        intra_class_ratio = within / (within + cross)
        return intra_class_ratio
