#!/usr/bin/env python3
"""
Backward Compatibility Layer for unified_preprocessing.py
This file provides imports from the actual unified_pipeline.py module
to maintain compatibility with legacy code that expects unified_preprocessing
"""

# Import everything from the actual unified_pipeline module
try:
    from scripts.unified_pipeline import *
except ImportError:
    try:
        from unified_pipeline import *
    except ImportError:
        print("❌ WARNING: unified_pipeline.py not found!")
        print("   Make sure scripts/unified_pipeline.py exists for the system to work properly.")
        
        # Provide minimal fallback to prevent complete failure
        class DummyPipeline:
            """Dummy pipeline to prevent import errors"""
            def __init__(self):
                self.is_ready = False
            
            def process_single_text(self, text):
                return {
                    'success': False,
                    'error': 'unified_pipeline not available',
                    'raw_text': text,
                    'processed_text': text,
                    'embedding': None
                }
        
        def get_global_pipeline(**kwargs):
            """Dummy function"""
            return DummyPipeline()
        
        def process_text_unified(text, **kwargs):
            """Dummy function"""
            return {
                'success': False,
                'error': 'unified_pipeline not available',
                'processed_text': text,
                'embedding': None
            }
        
        class UnifiedTextProcessor:
            """Dummy class"""
            def clean_text(self, text):
                return text
        
        class UnifiedPipelineProcessor:
            """Dummy class"""
            def __init__(self):
                self.is_ready = False
                


