from vocode.streaming.agent.base_agent import BaseAgent
from typing import Dict

class CallManager:
    agents_by_number: Dict[str, BaseAgent] = {}
