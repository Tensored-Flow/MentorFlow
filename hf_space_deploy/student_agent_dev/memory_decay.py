"""
Memory decay model using Ebbinghaus forgetting curve.

Scientific basis: Retention after time t: R(t) = exp(-t / τ)
where τ (tau) is the retention constant.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class MemoryRecord:
    """Record of practice session for a topic."""
    timestamp: float
    base_skill: float  # Skill level right after practice


class MemoryDecayModel:
    """
    Models realistic forgetting using Ebbinghaus curve.
    
    Key features:
    - Track last practice time per topic
    - Compute retention factor based on time elapsed
    - Effective skill = base_skill × retention_factor
    """
    
    def __init__(self, retention_constant: float = 80.0):
        """
        Args:
            retention_constant (tau): Controls forgetting speed.
                Higher = slower forgetting
                tau=80 means ~37% retention after 80 time steps
        """
        self.tau = retention_constant
        
        # Track per-topic memory
        self.topic_memories: Dict[str, MemoryRecord] = {}
        
        # Current time
        self.current_time: float = 0.0
    
    def update_practice(self, topic: str, base_skill: float):
        """
        Record that student just practiced a topic.
        
        Args:
            topic: Topic that was practiced
            base_skill: Student's skill level after practice (0.0-1.0)
        """
        self.topic_memories[topic] = MemoryRecord(
            timestamp=self.current_time,
            base_skill=base_skill
        )
    
    def get_retention_factor(self, topic: str) -> float:
        """
        Compute retention factor for a topic.
        
        Returns:
            Retention factor (0.0-1.0) based on Ebbinghaus curve
            1.0 = just practiced, decays exponentially over time
        """
        if topic not in self.topic_memories:
            return 1.0  # First time seeing topic
        
        memory = self.topic_memories[topic]
        time_elapsed = self.current_time - memory.timestamp
        
        # Ebbinghaus forgetting curve
        retention = np.exp(-time_elapsed / self.tau)
        
        return retention
    
    def get_effective_skill(self, topic: str) -> float:
        """
        Get current effective skill accounting for forgetting.
        
        Returns:
            Effective skill = base_skill × retention_factor
        """
        if topic not in self.topic_memories:
            return 0.0  # Never practiced
        
        memory = self.topic_memories[topic]
        retention = self.get_retention_factor(topic)
        
        return memory.base_skill * retention
    
    def get_time_since_practice(self, topic: str) -> float:
        """Get time elapsed since last practice."""
        if topic not in self.topic_memories:
            return float('inf')
        
        return self.current_time - self.topic_memories[topic].timestamp
    
    def advance_time(self, delta: float = 1.0):
        """Simulate time passing."""
        self.current_time += delta
    
    def get_all_topics(self) -> List[str]:
        """Get all topics that have been practiced."""
        return list(self.topic_memories.keys())
    
    def plot_forgetting_curves(self, topics: List[str] = None, 
                               save_path: str = 'forgetting_curves.png'):
        """
        Plot forgetting curves for topics.
        
        Shows how retention decays over time since last practice.
        """
        import matplotlib.pyplot as plt
        
        if topics is None:
            topics = self.get_all_topics()
        
        if not topics:
            print("⚠️ No topics to plot")
            return
        
        # Generate time points
        time_range = np.linspace(0, 200, 100)
        
        plt.figure(figsize=(10, 6))
        for topic in topics:
            retentions = [np.exp(-t / self.tau) for t in time_range]
            plt.plot(time_range, retentions, label=topic, linewidth=2)
        
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, 
                   label='50% retention threshold')
        plt.xlabel('Time Since Practice', fontsize=12)
        plt.ylabel('Retention Factor', fontsize=12)
        plt.title('Ebbinghaus Forgetting Curves', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"📊 Saved forgetting curves to {save_path}")

