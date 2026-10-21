import re
import nltk
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import logging

"""
NLP Processor Module for Resume Screening Bot
Handles text preprocessing, entity extraction, and embedding generation using scikit-learn
"""

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPProcessor:
    """
    NLP Processor for resume and job description analysis
    Uses scikit-learn for text processing and similarity computation
    """
    
    def __init__(self):
        """Initialize the NLP processor"""
        logger.info("Initializing NLP Processor with scikit-learn backend...")
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='ascii'
        )
        
        # Skills and keywords for NER
        self.tech_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'r', 'sql', 'html', 'css', 'php'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scipy'],
            'tools': ['excel', 'tableau', 'power bi', 'git', 'docker', 'kubernetes', 'aws', 'azure'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite'],
            'frameworks': ['django', 'flask', 'react', 'angular', 'spring', 'nodejs']
        }
        
        # Fitted flag
        self.is_fitted = False
        
        logger.info("NLP Processor initialized successfully!")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text
        
        Args:
            text: Text to process
            
        Returns:
            List of processed tokens
        """
        # Clean text first
        text = self.clean_text(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return processed_tokens
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract technical skills from text using NER
        
        Args:
            text: Text to extract skills from
            
        Returns:
            Dictionary of skill categories and found skills
        """
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.tech_skills.items():
            found = []
            for skill in skills:
                if skill in text_lower:
                    found.append(skill)
            if found:
                found_skills[category] = found
        
        return found_skills
    
    def extract_experience_years(self, text: str) -> Optional[int]:
        """
        Extract years of experience from text
        
        Args:
            text: Text to extract experience from
            
        Returns:
            Number of years of experience or None
        """
        # Patterns to match experience
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience'
        ]
        
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                return int(matches[0])
        
        return None
    
    def extract_education(self, text: str) -> List[str]:
        """
        Extract education information from text
        
        Args:
            text: Text to extract education from
            
        Returns:
            List of found education degrees
        """
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'mba', 'bs', 'ms', 'ma', 'ba',
            'computer science', 'engineering', 'mathematics', 'statistics', 'physics',
            'business administration', 'data science', 'analytics'
        ]
        
        text_lower = text.lower()
        found_education = []
        
        for keyword in education_keywords:
            if keyword in text_lower:
                found_education.append(keyword)
        
        return list(set(found_education))  # Remove duplicates
    
    def fit_vectorizer(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizer on the corpus
        
        Args:
            texts: List of texts to fit on
        """
        try:
            cleaned_texts = [self.clean_text(text) for text in texts]
            self.tfidf_vectorizer.fit(cleaned_texts)
            self.is_fitted = True
            logger.info("TF-IDF vectorizer fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting vectorizer: {e}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate TF-IDF embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of TF-IDF embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            cleaned_texts = [self.clean_text(text) for text in texts]
            
            if not self.is_fitted:
                # Fit and transform
                embeddings = self.tfidf_vectorizer.fit_transform(cleaned_texts)
                self.is_fitted = True
            else:
                # Just transform
                embeddings = self.tfidf_vectorizer.transform(cleaned_texts)
            
            return embeddings.toarray()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def get_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate TF-IDF embeddings for comparison
        
        Args:
            texts: List of texts to vectorize
            
        Returns:
            TF-IDF matrix
        """
        return self.get_embeddings(texts)
    
    def calculate_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embedding vectors
        
        Args:
            embeddings1: First embedding vector
            embeddings2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Ensure we have 2D arrays
            if embeddings1.ndim == 1:
                embeddings1 = embeddings1.reshape(1, -1)
            if embeddings2.ndim == 1:
                embeddings2 = embeddings2.reshape(1, -1)
            
            similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def analyze_text(self, text: str) -> Dict:
        """
        Comprehensive text analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        result = {
            'cleaned_text': self.clean_text(text),
            'tokens': self.tokenize_and_lemmatize(text),
            'skills': self.extract_skills(text),
            'experience_years': self.extract_experience_years(text),
            'education': self.extract_education(text),
            'word_count': len(text.split()),
            'character_count': len(text)
        }
        
        return result
    
    def process_document(self, text: str) -> Dict:
        """
        Process a document (resume or job description) and extract all relevant information
        
        Args:
            text: Document text to process
            
        Returns:
            Dictionary containing processed document data
        """
        try:
            # Get comprehensive analysis
            analysis = self.analyze_text(text)
            
            # Generate embeddings for the text
            embeddings = self.get_embeddings([text])
              # Combine all processed data
            processed_data = {
                'original_text': text,
                'cleaned_text': analysis['cleaned_text'],
                'tokens': analysis['tokens'],
                'skills': analysis['skills'],
                'skill_count': sum(len(skills) for skills in analysis['skills'].values()) if analysis['skills'] else 0,
                'experience_years': analysis['experience_years'],
                'education': analysis['education'],
                'word_count': analysis['word_count'],
                'character_count': analysis['character_count'],
                'embeddings': embeddings[0] if len(embeddings) > 0 else np.array([])
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise

    def get_feature_names(self) -> List[str]:
        """
        Get feature names from the fitted vectorizer
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            return []
        
        try:
            return self.tfidf_vectorizer.get_feature_names_out().tolist()
        except Exception as e:
            logger.error(f"Error getting feature names: {e}")
            return []
