from collections import Counter
from dataclasses import dataclass
from minisweagent.agents.default import AgentConfig, DefaultAgent, FormatError, LimitsExceeded
from sklearn.cluster import AgglomerativeClustering
from typing import List

import numpy as np

@dataclass
class ConsensusAgentConfig(AgentConfig):
    num_samples: int = 3
    use_embeddings: bool = False
    similarity_threshold: float = 0.7
    embedding_model: str = "microsoft/codebert-base"

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
    
    # def _vote_with_embeddings_cluster(self, candidates: List[tuple[str, dict]]) -> tuple[dict, int]:
    #     embedder = self._get_embedder()
    #     actions = [cand[0] for cand in candidates]
    #     embeddings = embedder.encode(actions, normalize_embeddings=True)

    #     # Agglomerative clustering (cosine distance)
    #     clustering = AgglomerativeClustering(
    #         n_clusters=None,
    #         distance_threshold=1 - self.config.similarity_threshold,
    #         metric="cosine",
    #         linkage="complete",
    #     )
    #     labels = clustering.fit_predict(embeddings)

    #     # Group samples by cluster id
    #     cluster_map = {}
    #     for idx, label in enumerate(labels):
    #         cluster_map.setdefault(label, []).append(idx)

    #     # Find the biggest cluster
    #     winning_label = max(cluster_map, key=lambda c: len(cluster_map[c]))
    #     cluster_indices = cluster_map[winning_label]

    #     # Calculate cluster centroid
    #     cluster_centroid = np.mean([embeddings[i] for i in cluster_indices], axis=0)
    #     cluster_centroid /= np.linalg.norm(cluster_centroid) + 1e-8

    #     # Pick the action closest to centroid
    #     winning_index = max(
    #         cluster_indices,
    #         key=lambda i: float(np.dot(embeddings[i], cluster_centroid))
    #     )

    #     voted_response = candidates[winning_index][1]
    #     count = len(cluster_indices)

    #     return voted_response, count

    def _vote_with_embeddings(self, candidates: List[tuple[str, dict]]) -> tuple[dict, int]:
        embedder = self._get_embedder()
        actions = [cand[0] for cand in candidates]
        embeddings = embedder.encode(actions, normalize_embeddings=True)

        embeddings = np.asarray(embeddings)
        sim_matrix = embeddings @ embeddings.T

        scores = sim_matrix.sum(axis=1)

        winning_index = int(np.argmax(scores))
        voted_response = candidates[winning_index][1]
        count = None

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

            try:
                self._embedder = SentenceTransformer(
                    self.config.embedding_model, device="cpu"
                )
            except NotImplementedError:
                # Some torch versions initialize parameters on the meta device when
                # low_cpu_mem_usage is enabled, which can fail when moving back to CPU.
                self._embedder = SentenceTransformer(
                    self.config.embedding_model,
                    device="cpu",
                    model_kwargs={"low_cpu_mem_usage": False},
                )

        return self._embedder