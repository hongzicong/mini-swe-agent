from collections import Counter
from dataclasses import dataclass
from minisweagent.agents.default import AgentConfig, DefaultAgent, FormatError, LimitsExceeded
# from sklearn.cluster import AgglomerativeClustering
from typing import List
import re
import numpy as np
import random

@dataclass
class ConsensusAgentConfig(AgentConfig):
    num_samples: int = 3
    voting_manner: str = "exact"
    similarity_threshold: float = 0.7
    # embedding_model: str = "microsoft/codebert-base"
    # embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    llm_judge_template: str = (
'''
Given the above output, you will be given multiple candidate actions. 
Each action is shown in a code block and labeled with an ID number.

Your goal: Select the single best action to execute next, based on the conversation and context so far.

**CRITICAL REQUIREMENTS:**
- Consider all candidate actions carefully.
- Choose exactly ONE action.

{{actions}}
'''
    )

#     llm_judge_template: str = (
# '''
# Given the above output, you will be given multiple candidate actions as follows.
# Each action is shown in a code block and labeled with an ID number.

# {{actions}}

# Your goal: Select the single best action to execute next, based on the conversation and context so far.

# **CRITICAL REQUIREMENTS:**
# - Consider all candidate actions carefully.
# - Choose exactly ONE action.

# Correct response format (example):
# <example_response>
# THOUGHT: I think Action X is the best because we need to first verify the content around the get_inline_instances method.
# </example_response>

# Where X is the chosen action ID (starting from 1).
# '''
#     )

class ConsensusAgent(DefaultAgent):

    def __init__(self, *args, config_class=ConsensusAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)

    def step(self) -> dict:
        valid_candidates = []
        all_responses = []

        for i in range(self.config.num_samples):
            if self.config.step_limit * self.config.num_samples <= self.model.n_calls or self.config.cost_limit * self.config.num_samples <= self.model.cost:
                raise LimitsExceeded()
            response = self.model.query(self.messages)
            all_responses.append(response)
            
            try:
                action_dict = self.parse_action(response)
                valid_candidates.append((action_dict['action'], response))
            except FormatError:
                continue
        
        voting_manner = self.config.voting_manner
        valid_voting = {"llm_judge", "embedding", "exact", "random"}
        if voting_manner not in valid_voting:
            raise ValueError(
                f"Unknown voting_manner '{voting_manner}'. Choose from {', '.join(sorted(valid_voting))}."
            )

        if len(valid_candidates) > 1:
            if voting_manner == "llm_judge":
                voted_response, consensus_details = self._vote_with_llm_judge(valid_candidates)
            elif voting_manner == "embedding":
                voted_response, consensus_details = self._vote_with_embeddings(valid_candidates)
            elif voting_manner == "random":
                voted_response, consensus_details = self._vote_random(valid_candidates)
            elif voting_manner == "exact":
                voted_response, consensus_details = self._vote_exact(valid_candidates)
        elif len(valid_candidates) == 1:
            voted_response, consensus_details = valid_candidates[0][1], 1
        else:
            self.parse_action(all_responses[0])

        self.add_message(
            "assistant",
            **voted_response,
            consensus_info={
                "consensus_details": consensus_details,
                "consensus_candidates": [response["content"] for response in all_responses],
            },
        )

        return self.get_observation(voted_response)
    
    # # Clustering-based Voting
    # def _vote_with_embeddings(self, candidates: List[tuple[str, dict]]) -> tuple[dict, int]:
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

    # Pairwise Similarity-based Voting
    def _vote_with_embeddings(self, candidates: List[tuple[str, dict]]) -> tuple[dict, int]:
        embedder = self._get_embedder()
        actions = [cand[0] for cand in candidates]
        embeddings = embedder.encode(actions, normalize_embeddings=True)

        embeddings = np.asarray(embeddings)
        sim_matrix = embeddings @ embeddings.T

        scores = sim_matrix.sum(axis=1)

        winning_index = int(np.argmax(scores))
        voted_response = candidates[winning_index][1]

        return voted_response, scores.tolist()
    
    def _vote_with_llm_judge(self, candidates: List[tuple[str, dict]]) -> tuple[dict, int | None]:
        formatted_actions = [
            f"Action {idx}:\n```bash\n{action}\n```"
            for idx, (action, _) in enumerate(candidates, start=1)
        ]
        
        prompt = self.render_template(
            self.config.llm_judge_template, actions="\n\n".join(formatted_actions)
        )
        judge_messages = self.messages[:-1] + [{"role": "user", "content": self.messages[-1]["content"] + prompt}]

        if self.config.step_limit * self.config.num_samples <= self.model.n_calls or self.config.cost_limit * self.config.num_samples <= self.model.cost:
            raise LimitsExceeded()

        judge_response = self.model.query(judge_messages)

        choice_match = re.findall(r"(?:Action\s*)?(\d+)", judge_response["content"])
        winning_index = int(choice_match[0]) - 1 if choice_match else 0
        if not 0 <= winning_index < len(candidates):
            winning_index = 0

        voted_response = candidates[winning_index][1]

        return voted_response, {"judge_response": judge_response["content"], "winning_index": winning_index}
    
    def _vote_exact(self, candidates: List[tuple[str, dict]]) -> tuple[dict, int]:
        actions = [cand[0] for cand in candidates]
        counter = Counter(actions)
        winner_action, count = counter.most_common(1)[0]

        voted_response = next(resp for act, resp in candidates if act == winner_action)

        return voted_response, count
    
    def _vote_random(self, candidates: List[tuple[str, dict]]) -> tuple[dict, int]:
        winning_index = random.randint(0, len(candidates) - 1)
        voted_response = candidates[winning_index][1]

        return voted_response, winning_index

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