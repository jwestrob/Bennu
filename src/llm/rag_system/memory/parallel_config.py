"""
Configuration for parallel task execution.

Provides easy tuning of parallel execution parameters for optimal performance
based on your system capabilities and API rate limits.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class ParallelExecutionProfile:
    """Predefined execution profiles for different scenarios."""
    name: str
    max_concurrent_tasks: int
    batch_size: int
    timeout_per_task: float
    retry_failed_tasks: bool
    max_retries: int
    description: str

# Predefined execution profiles
EXECUTION_PROFILES = {
    "conservative": ParallelExecutionProfile(
        name="conservative",
        max_concurrent_tasks=5,
        batch_size=3,
        timeout_per_task=180.0,  # 3 minutes
        retry_failed_tasks=True,
        max_retries=3,
        description="Safe settings for API rate limit compliance"
    ),
    
    "balanced": ParallelExecutionProfile(
        name="balanced", 
        max_concurrent_tasks=10,
        batch_size=5,
        timeout_per_task=120.0,  # 2 minutes
        retry_failed_tasks=True,
        max_retries=2,
        description="Good balance of speed and reliability (recommended)"
    ),
    
    "aggressive": ParallelExecutionProfile(
        name="aggressive",
        max_concurrent_tasks=20,
        batch_size=10,
        timeout_per_task=90.0,  # 1.5 minutes
        retry_failed_tasks=True,
        max_retries=1,
        description="Maximum speed - may hit API rate limits"
    ),
    
    "ultra": ParallelExecutionProfile(
        name="ultra",
        max_concurrent_tasks=50,
        batch_size=25,
        timeout_per_task=60.0,  # 1 minute
        retry_failed_tasks=False,
        max_retries=0,
        description="Extreme speed - for powerful systems and unlimited APIs"
    )
}

class ParallelConfigManager:
    """Manages parallel execution configuration with easy switching."""
    
    def __init__(self):
        self.current_profile = "balanced"
        self.custom_config: Optional[ParallelExecutionProfile] = None
    
    def set_profile(self, profile_name: str) -> bool:
        """
        Set execution profile by name.
        
        Args:
            profile_name: Name of profile ('conservative', 'balanced', 'aggressive', 'ultra')
            
        Returns:
            True if profile was set successfully
        """
        if profile_name in EXECUTION_PROFILES:
            self.current_profile = profile_name
            self.custom_config = None
            logger.info(f"🎯 Switched to '{profile_name}' execution profile")
            logger.info(f"   {EXECUTION_PROFILES[profile_name].description}")
            return True
        else:
            logger.error(f"❌ Unknown profile: {profile_name}")
            logger.info(f"   Available profiles: {list(EXECUTION_PROFILES.keys())}")
            return False
    
    def set_custom_config(self, 
                         max_concurrent: int = 10,
                         batch_size: int = 5,
                         timeout: float = 120.0,
                         retries: int = 2) -> None:
        """
        Set custom execution configuration.
        
        Args:
            max_concurrent: Maximum concurrent API calls
            batch_size: Number of tasks per batch
            timeout: Timeout per task in seconds
            retries: Maximum retry attempts
        """
        self.custom_config = ParallelExecutionProfile(
            name="custom",
            max_concurrent_tasks=max_concurrent,
            batch_size=batch_size,
            timeout_per_task=timeout,
            retry_failed_tasks=retries > 0,
            max_retries=retries,
            description=f"Custom: {max_concurrent} concurrent, {batch_size} batch size"
        )
        self.current_profile = "custom"
        logger.info(f"🎯 Set custom execution profile: {self.custom_config.description}")
    
    def get_current_config(self) -> ParallelExecutionProfile:
        """Get current execution configuration."""
        if self.custom_config:
            return self.custom_config
        return EXECUTION_PROFILES[self.current_profile]
    
    def print_current_config(self) -> None:
        """Print current configuration details."""
        config = self.get_current_config()
        print(f"\\n🎯 Current Parallel Execution Config: {config.name}")
        print(f"   Description: {config.description}")
        print(f"   Max concurrent tasks: {config.max_concurrent_tasks}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Timeout per task: {config.timeout_per_task}s")
        print(f"   Retry failed tasks: {config.retry_failed_tasks}")
        print(f"   Max retries: {config.max_retries}")
    
    def print_all_profiles(self) -> None:
        """Print all available execution profiles."""
        print("\\n📊 Available Execution Profiles:")
        for name, profile in EXECUTION_PROFILES.items():
            current_marker = " (CURRENT)" if name == self.current_profile else ""
            print(f"   {name}{current_marker}: {profile.description}")
            print(f"      {profile.max_concurrent_tasks} concurrent, {profile.batch_size} batch, {profile.timeout_per_task}s timeout")
    
    def estimate_speedup(self, num_tasks: int) -> dict:
        """
        Estimate execution time and speedup for given number of tasks.
        
        Args:
            num_tasks: Number of tasks to execute
            
        Returns:
            Dictionary with time estimates
        """
        config = self.get_current_config()
        
        # Estimate sequential time (assume 10s per task)
        sequential_time = num_tasks * 10
        
        # Estimate parallel time
        batches = (num_tasks + config.batch_size - 1) // config.batch_size
        parallel_time_per_batch = config.batch_size * 10 / config.max_concurrent_tasks
        parallel_time = batches * parallel_time_per_batch
        
        # Add overhead and batch delays
        overhead = batches * 0.5  # 0.5s delay between batches
        parallel_time += overhead
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1
        
        return {
            'num_tasks': num_tasks,
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'time_saved': sequential_time - parallel_time
        }

# Global config manager
_config_manager = ParallelConfigManager()

def set_parallel_profile(profile_name: str) -> bool:
    """Set parallel execution profile globally."""
    return _config_manager.set_profile(profile_name)

def set_custom_parallel_config(max_concurrent: int = 10, 
                              batch_size: int = 5,
                              timeout: float = 120.0,
                              retries: int = 2) -> None:
    """Set custom parallel execution configuration globally.""" 
    _config_manager.set_custom_config(max_concurrent, batch_size, timeout, retries)

def get_parallel_config() -> ParallelExecutionProfile:
    """Get current parallel execution configuration."""
    return _config_manager.get_current_config()

def print_parallel_status() -> None:
    """Print current parallel execution status."""
    _config_manager.print_current_config()

def print_parallel_profiles() -> None:
    """Print all available parallel execution profiles."""
    _config_manager.print_all_profiles()

def estimate_parallel_speedup(num_tasks: int) -> dict:
    """Estimate speedup for parallel execution."""
    return _config_manager.estimate_speedup(num_tasks)

# Convenience functions
def set_conservative_parallel():
    """Switch to conservative parallel execution (safe for API limits)."""
    set_parallel_profile("conservative")

def set_balanced_parallel():
    """Switch to balanced parallel execution (recommended default)."""
    set_parallel_profile("balanced")

def set_aggressive_parallel():
    """Switch to aggressive parallel execution (maximum speed)."""
    set_parallel_profile("aggressive")

def set_ultra_parallel():
    """Switch to ultra parallel execution (extreme speed)."""
    set_parallel_profile("ultra")