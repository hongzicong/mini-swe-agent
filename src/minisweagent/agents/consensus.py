from collections import Counter
from dataclasses import dataclass
from minisweagent.agents.default import AgentConfig, DefaultAgent, FormatError, LimitsExceeded
from typing import List

import numpy as np

@dataclass
class ConsensusAgentConfig(AgentConfig):
    num_samples: int = 3
    use_embeddings: bool = False
    similarity_threshold: float = 0.7
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

class ConsensusAgent(DefaultAgent):

    def __init__(self, *args, config_class=ConsensusAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)

    def step(self) -> dict:
        valid_candidates = []
        all_responses = []

        for _ in range(self.config.num_samples):
            if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
                raise LimitsExceeded()
            response = self.model.query(self.messages)
            all_responses.append(response)
            
            try:
                action_dict = self.parse_action(response)
                valid_candidates.append((action_dict['action'], response))
            except FormatError:
                continue
        
        if not valid_candidates:
            self.parse_action(all_responses[0])
        
        if self.config.use_embeddings and len(valid_candidates) > 1:
            voted_response, count = self._vote_with_embeddings(valid_candidates)
        else:
            actions = [cand[0] for cand in valid_candidates]
            counter = Counter(actions)
            winner_action, count = counter.most_common(1)[0]

            voted_response = next(resp for act, resp in valid_candidates if act == winner_action)

        self.add_message("assistant", **voted_response, consensus_count=count)

        return self.get_observation(voted_response)
    
    def _vote_with_embeddings(self, candidates: List[tuple[str, dict]]) -> tuple[dict, int]:
        embedder = self._get_embedder()
        actions = [cand[0] for cand in candidates]
        embeddings = embedder.encode(actions, normalize_embeddings=True)

        clusters = []
        for idx, embedding in enumerate(embeddings):
            embedding_vector = np.array(embedding, dtype=np.float64)
            best_cluster = None
            best_similarity = 0.0

            for cluster in clusters:
                # dot product == cosine similarity (because normalized)
                similarity = float(np.dot(embedding_vector, cluster["centroid"]))

                # --- Step 1: use a lower threshold to allow clustering ---
                if similarity >= self.config.similarity_threshold and similarity > best_similarity:
                    best_cluster = cluster
                    best_similarity = similarity

            # Create new cluster if no appropriate cluster found
            if best_cluster is None:
                clusters.append({
                    "indices": [idx],
                    "sum": embedding_vector,
                    "centroid": embedding_vector,
                })
            else:
                # add to existing cluster
                best_cluster["indices"].append(idx)
                best_cluster["sum"] += embedding_vector
                norm = np.linalg.norm(best_cluster["sum"])
                if norm == 0:
                    best_cluster["centroid"] = best_cluster["sum"]
                else:
                    best_cluster["centroid"] = best_cluster["sum"] / norm

        # --- Find the cluster with most members ---
        winning_cluster = max(clusters, key=lambda c: len(c["indices"]))
        cluster_indices = winning_cluster["indices"]
        winning_index = winning_cluster["indices"][0] 
        voted_response = candidates[winning_index][1] 

        count = len(cluster_indices)

        return voted_response, count

    def _get_embedder(self):
        if not hasattr(self, "_embedder"):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - handled by configuration
                raise ImportError(
                    "sentence-transformers is required for embedding-based consensus;"
                    " install it or disable embeddings."
                ) from exc

            self._embedder = SentenceTransformer(self.config.embedding_model)

        return self._embedder