from collections import Counter
from dataclasses import dataclass
from minisweagent.agents.default import AgentConfig, DefaultAgent, FormatError, LimitsExceeded


@dataclass
class ConsensusAgentConfig(AgentConfig):
    num_samples: int = 3
    

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
        
        actions = [cand[0] for cand in valid_candidates]
        counter = Counter(actions)
        winner_action, count = counter.most_common(1)[0]

        voted_response = next(resp for act, resp in valid_candidates if act == winner_action)

        self.add_message("assistant", **voted_response, consensus_count=count)

        return self.get_observation(voted_response)