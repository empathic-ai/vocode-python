from vocode.streaming.agent.base_agent import BaseAgent

class CallManager:
    agents_by_number: Dict[str, BaseAgent] = {}
