#!/usr/bin/env python3
"""
Unified Preprocessing Module
Ensures IDENTICAL preprocessing between training and inference pipelines
Extracted from preprocess.py and embed_dataset.py for consistency
"""

import re
import string
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

class UnifiedTextProcessor:
    """
    Unified text processor that ensures IDENTICAL preprocessing
    between training pipeline and GUI/inference pipeline
    """
    
    def __init__(self):
        """Initialize processor with same logic as preprocess.py"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Stopwords list (same as in preprocess.py)
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
    
    def clean_text(self, text: str) -> str:
        """
        IDENTICAL text cleaning as preprocess.py
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords (same logic as preprocess.py)
        
        Args:
            text: Cleaned text
            
        Returns:
            Text without stopwords
        """
        if not text:
            return ""
        
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def preprocess_text(self, text: str, remove_stopwords: bool = True) -> str:
        """
        Complete preprocessing pipeline (IDENTICAL to preprocess.py)
        
        Args:
            text: Raw text to preprocess
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Fully preprocessed text
        """
        if not text:
            return ""
        
        # Step 1: Clean text
        cleaned = self.clean_text(text)
        
        # Step 2: Remove stopwords if requested
        if remove_stopwords:
            cleaned = self.remove_stopwords(cleaned)
        
        # Step 3: Final validation
        if len(cleaned.strip()) < 3:  # Too short after cleaning
            return ""
        
        return cleaned.strip()
    
    def preprocess_batch(self, texts: List[str], remove_stopwords: bool = True) -> List[str]:
        """
        Preprocess batch of texts
        
        Args:
            texts: List of raw texts
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            List of preprocessed texts
        """
        processed = []
        for text in texts:
            processed_text = self.preprocess_text(text, remove_stopwords)
            processed.append(processed_text)
        
        return processed

class UnifiedEmbeddingGenerator:
    """
    Unified embedding generator that ensures IDENTICAL embeddings
    between training pipeline (embed_dataset.py) and inference pipeline
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize embedding generator
        
        Args:
            model_path: Path to embedding model (same as embed_dataset.py)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.model_path = model_path
        
        # Default model paths (same priority as embed_dataset.py)
        self.default_model_paths = [
            "models/minilm-l6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2"
        ]
        
        self._load_model()
    
    def _load_model(self):
        """Load embedding model (same logic as embed_dataset.py)"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self.logger.error("SentenceTransformers not available")
            return False
        
        # Try specified model path first
        if self.model_path:
            try:
                local_path = Path(self.model_path)
                if local_path.exists():
                    self.model = SentenceTransformer(str(local_path))
                    self.logger.info(f"✅ Loaded local model: {local_path}")
                    return True
                else:
                    self.model = SentenceTransformer(self.model_path)
                    self.logger.info(f"✅ Loaded online model: {self.model_path}")
                    return True
            except Exception as e:
                self.logger.warning(f"Failed to load specified model {self.model_path}: {e}")
        
        # Try default models
        for model_name in self.default_model_paths:
            try:
                local_path = Path(model_name)
                if local_path.exists() and local_path.is_dir():
                    self.model = SentenceTransformer(str(local_path))
                    self.logger.info(f"✅ Loaded local model: {local_path}")
                    return True
                
                self.model = SentenceTransformer(model_name)
                self.logger.info(f"✅ Loaded online model: {model_name}")
                return True
                
            except Exception as e:
                self.logger.debug(f"Failed to load {model_name}: {e}")
                continue
        
        self.logger.error("❌ Could not load any embedding model")
        return False
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for single text (IDENTICAL to embed_dataset.py logic)
        
        Args:
            text: Preprocessed text
            
        Returns:
            Embedding vector or None if failed
        """
        if not self.model:
            self.logger.error("Embedding model not available")
            return None
        
        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return None
        
        try:
            # Use same parameters as embed_dataset.py
            embedding = self.model.encode(
                [text], 
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=1
            )
            return embedding[0]
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for batch of texts (same as embed_dataset.py)
        
        Args:
            texts: List of preprocessed texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            self.logger.error("Embedding model not available")
            return [None] * len(texts)
        
        try:
            # Filter out empty texts but keep track of indices
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                return [None] * len(texts)
            
            # Generate embeddings (same parameters as embed_dataset.py)
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=len(valid_texts) > 10,
                batch_size=batch_size
            )
            
            # Map back to original indices
            result = [None] * len(texts)
            for valid_idx, original_idx in enumerate(valid_indices):
                result[original_idx] = embeddings[valid_idx]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {e}")
            return [None] * len(texts)

class UnifiedPipelineProcessor:
    """
    Complete unified pipeline processor that ensures IDENTICAL processing
    between training and inference
    """
    
    def __init__(self, embedding_model_path: str = None, remove_stopwords: bool = True):
        """
        Initialize unified pipeline
        
        Args:
            embedding_model_path: Path to embedding model
            remove_stopwords: Whether to remove stopwords in preprocessing
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.text_processor = UnifiedTextProcessor()
        self.embedding_generator = UnifiedEmbeddingGenerator(embedding_model_path)
        self.remove_stopwords = remove_stopwords
        
        # Validate initialization
        self.is_ready = self.embedding_generator.model is not None
        
        if self.is_ready:
            self.logger.info("✅ Unified pipeline ready")
        else:
            self.logger.error("❌ Unified pipeline initialization failed")
    
    def process_single_text(self, raw_text: str) -> Dict[str, Any]:
        """
        Process single text through COMPLETE pipeline
        (preprocessing + embedding)
        
        Args:
            raw_text: Raw input text
            
        Returns:
            Dict with processed text, embedding, and metadata
        """
        if not self.is_ready:
            return {
                'success': False,
                'error': 'Pipeline not ready - embedding model unavailable',
                'raw_text': raw_text,
                'processed_text': '',
                'embedding': None
            }
        
        try:
            # Step 1: Preprocess text (IDENTICAL to preprocess.py)
            processed_text = self.text_processor.preprocess_text(
                raw_text, 
                remove_stopwords=self.remove_stopwords
            )
            
            if not processed_text:
                return {
                    'success': False,
                    'error': 'Text became empty after preprocessing',
                    'raw_text': raw_text,
                    'processed_text': '',
                    'embedding': None
                }
            
            # Step 2: Generate embedding (IDENTICAL to embed_dataset.py)
            embedding = self.embedding_generator.embed_text(processed_text)
            
            if embedding is None:
                return {
                    'success': False,
                    'error': 'Embedding generation failed',
                    'raw_text': raw_text,
                    'processed_text': processed_text,
                    'embedding': None
                }
            
            return {
                'success': True,
                'raw_text': raw_text,
                'processed_text': processed_text,
                'embedding': embedding,
                'embedding_shape': embedding.shape,
                'text_length_raw': len(raw_text),
                'text_length_processed': len(processed_text)
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            return {
                'success': False,
                'error': f'Pipeline error: {str(e)}',
                'raw_text': raw_text,
                'processed_text': '',
                'embedding': None
            }
    
    def process_batch(self, raw_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process batch of texts through complete pipeline
        
        Args:
            raw_texts: List of raw input texts
            
        Returns:
            List of processing results
        """
        if not self.is_ready:
            error_result = {
                'success': False,
                'error': 'Pipeline not ready',
                'processed_text': '',
                'embedding': None
            }
            return [error_result] * len(raw_texts)
        
        try:
            # Step 1: Preprocess all texts
            processed_texts = self.text_processor.preprocess_batch(
                raw_texts,
                remove_stopwords=self.remove_stopwords
            )
            
            # Step 2: Generate embeddings for all texts
            embeddings = self.embedding_generator.embed_batch(processed_texts)
            
            # Step 3: Combine results
            results = []
            for i, raw_text in enumerate(raw_texts):
                processed_text = processed_texts[i]
                embedding = embeddings[i]
                
                if processed_text and embedding is not None:
                    result = {
                        'success': True,
                        'raw_text': raw_text,
                        'processed_text': processed_text,
                        'embedding': embedding,
                        'embedding_shape': embedding.shape,
                        'text_length_raw': len(raw_text),
                        'text_length_processed': len(processed_text)
                    }
                else:
                    result = {
                        'success': False,
                        'error': 'Preprocessing or embedding failed',
                        'raw_text': raw_text,
                        'processed_text': processed_text,
                        'embedding': None
                    }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch pipeline processing failed: {e}")
            error_result = {
                'success': False,
                'error': f'Batch pipeline error: {str(e)}',
                'processed_text': '',
                'embedding': None
            }
            return [error_result] * len(raw_texts)

# Global pipeline instance for reuse
_global_pipeline = None

def get_global_pipeline(embedding_model_path: str = None, 
                       remove_stopwords: bool = True) -> UnifiedPipelineProcessor:
    """
    Get or create global pipeline instance for efficient reuse
    
    Args:
        embedding_model_path: Path to embedding model
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        UnifiedPipelineProcessor instance
    """
    global _global_pipeline
    
    if _global_pipeline is None:
        _global_pipeline = UnifiedPipelineProcessor(
            embedding_model_path=embedding_model_path,
            remove_stopwords=remove_stopwords
        )
    
    return _global_pipeline

def process_text_unified(raw_text: str, 
                        embedding_model_path: str = None,
                        remove_stopwords: bool = True) -> Dict[str, Any]:
    """
    Convenience function for processing single text with unified pipeline
    
    Args:
        raw_text: Raw input text
        embedding_model_path: Path to embedding model
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Processing result dict
    """
    pipeline = get_global_pipeline(embedding_model_path, remove_stopwords)
    return pipeline.process_single_text(raw_text)

def main():
    """Test unified preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Unified Preprocessing Pipeline")
    parser.add_argument('--text', '-t', default="This is a great movie! I loved it.",
                        help='Text to process')
    parser.add_argument('--model', '-m', help='Embedding model path')
    parser.add_argument('--no-stopwords', action='store_true',
                        help='Keep stopwords')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing Unified Preprocessing Pipeline")
    print(f"Input text: {args.text}")
    print()
    
    # Test processing
    result = process_text_unified(
        args.text,
        embedding_model_path=args.model,
        remove_stopwords=not args.no_stopwords
    )
    
    if result['success']:
        print("✅ Processing successful!")
        print(f"📝 Raw text: {result['raw_text']}")
        print(f"🧹 Processed text: {result['processed_text']}")
        print(f"📊 Embedding shape: {result['embedding_shape']}")
        print(f"📏 Length raw: {result['text_length_raw']}")
        print(f"📏 Length processed: {result['text_length_processed']}")
    else:
        print("❌ Processing failed!")
        print(f"Error: {result['error']}")
    
    return 0 if result['success'] else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
