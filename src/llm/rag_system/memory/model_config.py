"""
Configuration interface for easy model switching in multi-part reports.

Provides simple functions to switch between cost-optimized and premium model modes.
"""

import logging
from typing import Dict, Any

from .model_allocation import get_model_allocator

logger = logging.getLogger(__name__)


class ModelConfigManager:
    """
    Simple interface for managing model configurations.
    
    Provides easy switching between different model modes for testing and production.
    """
    
    def __init__(self):
        """Initialize model configuration manager."""
        self.allocator = get_model_allocator()
        self.current_mode = "optimized"  # Default to cost-optimized
    
    def set_optimized_mode(self):
        """
        Set to cost-optimized mode.
        
        Uses gpt-4.1-mini for most tasks, o3 only for complex synthesis.
        Estimated cost savings: 70-80% compared to premium everywhere.
        """
        self.allocator.switch_to_optimized_mode()
        self.current_mode = "optimized"
        logger.info("🎯 Switched to OPTIMIZED mode: gpt-4.1-mini for most tasks, o3 for synthesis")
        
        # Print allocation summary
        summary = self.allocator.get_allocation_summary()
        if summary['mode'] == 'optimized':
            logger.info(f"💰 Estimated cost savings: {summary['cost_savings_vs_premium']}")
    
    def set_premium_mode(self):
        """
        Set to premium mode.
        
        Uses o3 for all tasks. Maximum quality but highest cost.
        """
        self.allocator.switch_to_premium_mode()
        self.current_mode = "premium"
        logger.info("🔥 Switched to PREMIUM mode: o3 for all tasks (maximum quality, highest cost)")
    
    def set_testing_mode(self):
        """
        Set to testing mode.
        
        Uses cheaper models for initial testing to avoid expensive API calls.
        """
        self.allocator.switch_to_optimized_mode()
        self.current_mode = "testing"
        logger.info("🧪 Switched to TESTING mode: using cheaper models for development")
    
    def get_current_mode(self) -> str:
        """Get the current mode."""
        return self.current_mode
    
    def get_status(self) -> Dict[str, Any]:
        """Get current configuration status."""
        summary = self.allocator.get_allocation_summary()
        
        status = {
            "current_mode": self.current_mode,
            "allocation_mode": summary['mode'],
            "status": "active"
        }
        
        if summary['mode'] == 'optimized':
            status["cost_savings"] = summary['cost_savings_vs_premium']
            status["primary_models"] = {
                "simple_tasks": "gpt-4.1-mini",
                "complex_tasks": "o3"
            }
        else:
            status["primary_model"] = "o3"
            status["cost_multiplier"] = "100x baseline"
        
        return status
    
    def print_allocation_details(self):
        """Print detailed allocation information."""
        summary = self.allocator.get_allocation_summary()
        
        print(f"\n🎯 Current Model Allocation Status:")
        print(f"Mode: {self.current_mode.upper()}")
        print(f"Allocation: {summary['mode']}")
        
        if summary['mode'] == 'optimized':
            print(f"Cost Savings: {summary['cost_savings_vs_premium']}")
            print(f"\nTask Allocation:")
            for task, config in summary['task_allocation'].items():
                print(f"  {task}: {config['model']} ({config['complexity']})")
        else:
            print(f"Primary Model: {summary['primary_model']}")
            print(f"Cost Impact: {summary['estimated_cost_multiplier']}x more expensive")
        
        print()


# Global configuration manager
config_manager = ModelConfigManager()


def set_optimized_mode():
    """Switch to cost-optimized mode globally."""
    config_manager.set_optimized_mode()


def set_premium_mode():
    """Switch to premium mode globally."""
    config_manager.set_premium_mode()


def set_testing_mode():
    """Switch to testing mode globally."""
    config_manager.set_testing_mode()


def get_current_mode() -> str:
    """Get current mode."""
    return config_manager.get_current_mode()


def print_model_status():
    """Print current model allocation status."""
    config_manager.print_allocation_details()


def get_config_manager() -> ModelConfigManager:
    """Get the global configuration manager."""
    return config_manager


# Convenience functions for quick switching
def quick_switch_to_o3():
    """Quick switch to o3 everywhere - for when you need maximum quality."""
    print("🔥 Switching to o3 everywhere for maximum quality...")
    set_premium_mode()
    print_model_status()


def quick_switch_to_optimized():
    """Quick switch to optimized mode - for cost-effective analysis."""
    print("🎯 Switching to optimized mode for cost-effective analysis...")
    set_optimized_mode()
    print_model_status()


def quick_switch_to_testing():
    """Quick switch to testing mode - for development and debugging."""
    print("🧪 Switching to testing mode for development...")
    set_testing_mode()
    print_model_status()


# Example usage functions
def demo_model_switching():
    """Demonstrate model switching capabilities."""
    print("=== Model Switching Demo ===")
    
    print("\n1. Starting in optimized mode:")
    quick_switch_to_optimized()
    
    print("\n2. Switching to premium mode:")
    quick_switch_to_o3()
    
    print("\n3. Switching to testing mode:")
    quick_switch_to_testing()
    
    print("\n4. Back to optimized mode:")
    quick_switch_to_optimized()
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_model_switching()