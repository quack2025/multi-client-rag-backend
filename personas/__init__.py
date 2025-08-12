# personas/__init__.py
"""
Comprehensive Persona System
Multi-client synthetic user generation and interaction system
"""

from .persona_system import ComprehensivePersonaSystem
from .persona_characteristics import EthicalPersonaGenerator
from .role_prompting_engine import RolePromptingEngine
from .bias_detection import BiasDetectionFramework
from .context_rich_prompting import ContextRichPromptGenerator
from .temperature_optimization import AdvancedTemperatureController
from .implicit_demographics import HondurasImplicitDemographics
from .temporal_context import HondurasTemporalContextManager
from .staged_validation import StagedPersonaValidator

__all__ = [
    'ComprehensivePersonaSystem',
    'EthicalPersonaGenerator',
    'RolePromptingEngine',
    'BiasDetectionFramework',
    'ContextRichPromptGenerator',
    'AdvancedTemperatureController',
    'HondurasImplicitDemographics',
    'HondurasTemporalContextManager',
    'StagedPersonaValidator'
]