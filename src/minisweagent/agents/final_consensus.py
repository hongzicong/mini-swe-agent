from collections import Counter
from dataclasses import asdict, dataclass
from minisweagent.agents.default import DefaultAgent, AgentConfig
from minisweagent.agents.consensus import ConsensusAgent, ConsensusAgentConfig
from typing import List

import numpy as np

@dataclass
class FinalConsensusAgentConfig(ConsensusAgentConfig):
    pass

class FinalConsensusAgent(ConsensusAgent):

    def __init__(self, *args, config_class=ConsensusAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run multiple independent agent trajectories and vote on final outputs."""

        base_config_fields = AgentConfig.__dataclass_fields__.keys()
        base_config_kwargs = {
            key: value for key, value in asdict(self.config).items() if key in base_config_fields
        }

        samples: list[tuple[str, str, list[dict[str, str]]]] = []
        for _ in range(self.config.num_samples):
            agent = DefaultAgent(
                self.model,
                self.env,
                config_class=AgentConfig,
                **base_config_kwargs,
            )
            status, result = agent.run(task, **kwargs)
            samples.append((status, result, agent.messages))

        valid_candidates: list[tuple[str, list[dict[str, str]]]] = [
            (result, messages) for status, result, messages in samples if status == "Submitted"
        ]

        if not valid_candidates:
            status, result, messages = samples[0]
            self.messages = messages
            return status, result

        if self.config.use_embeddings and len(valid_candidates) > 1:
            voted_message, count, voted_messages = self._vote_with_embeddings_on_text(valid_candidates)
        else:
            final_outputs = [candidate[0] for candidate in valid_candidates]
            counter = Counter(final_outputs)
            voted_message, count = counter.most_common(1)[0]
            voted_messages = next(messages for text, messages in valid_candidates if text == voted_message)

        self.messages = voted_messages

        return "Submitted", voted_message

    def _vote_with_embeddings_on_text(
        self, candidates: List[tuple[str, list[dict[str, str]]]]
    ) -> tuple[str, int | None, list[dict[str, str]]]:
        embedder = self._get_embedder()
        texts = [text for text, _ in candidates]
        embeddings = embedder.encode(texts, normalize_embeddings=True)
        embeddings = np.asarray(embeddings)
        sim_matrix = embeddings @ embeddings.T
        scores = sim_matrix.sum(axis=1)

        winning_index = int(np.argmax(scores))
        voted_message = candidates[winning_index][0]
        voted_messages = candidates[winning_index][1]
        count = None

        return voted_message, count, voted_messages