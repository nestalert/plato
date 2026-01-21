import re
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class NLPProcessor:
    
    def __init__(self, intents, model_file='chatbot_model.pkl', feedback_file='feedback_data.pkl'):
        self.intents = intents
        self.model_file = model_file
        self.feedback_file = feedback_file
        self.vectorizer = None
        self.pattern_vectors = None
        self.pattern_tags = []
        self.pattern_lookup = {}
        self.tag_to_intent = {}
        self.feedback_data = {}
        self.feedback_vectors = None
        self.feedback_tags = []
        self.feedback_weights = []
        
        self._build_pattern_lookup()
    
    def _build_pattern_lookup(self):
        self.pattern_lookup = {}
        for intent in self.intents:
            tag = intent['tag']
            for pattern in intent['patterns']:
                clean_pattern = pattern.lower().strip()
                self.pattern_lookup[clean_pattern] = tag
                if clean_pattern.endswith('?'):
                    self.pattern_lookup[clean_pattern.rstrip('?')] = tag
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()

        if len(words) < 4:
            return " ".join(words)

        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        
        important_words = {'what', 'where', 'when', 'how', 'who', 'why'}
        
        filtered_words = []
        for w in words:
            if w in important_words or w not in stop_words:
                filtered_words.append(stemmer.stem(w))
        
        if not filtered_words:
            return " ".join([stemmer.stem(w) for w in words])

        return " ".join(filtered_words)
    
    def prepare_training_data(self):
        patterns = []
        tags = []
        
        for intent in self.intents:
            tag = intent['tag']
            self.tag_to_intent[tag] = intent
            
            for pattern in intent['patterns']:
                patterns.append(self.preprocess_text(pattern))
                tags.append(tag)
        
        return patterns, tags
    
    def train(self):
        patterns, tags = self.prepare_training_data()
        
        self.vectorizer = TfidfVectorizer()
        self.pattern_vectors = self.vectorizer.fit_transform(patterns)
        self.pattern_tags = tags
        
        self.update_feedback_vectors()
    
    def predict_intent(self, user_input):
        clean_input = user_input.lower().strip()
        
        if clean_input in self.pattern_lookup:
            return self.pattern_lookup[clean_input], 1.0
            
        clean_input_no_punct = clean_input.rstrip('?.!')
        if clean_input_no_punct in self.pattern_lookup:
            return self.pattern_lookup[clean_input_no_punct], 1.0

        processed_input = self.preprocess_text(user_input)
        input_vector = self.vectorizer.transform([processed_input])
        
        similarities = cosine_similarity(input_vector, self.pattern_vectors)[0]
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        predicted_tag = self.pattern_tags[best_match_idx]
        
        if self.feedback_vectors is not None and len(self.feedback_tags) > 0:
            feedback_similarities = cosine_similarity(input_vector, self.feedback_vectors)[0]
            weighted_feedback = feedback_similarities * np.array(self.feedback_weights)
            best_feedback_idx = np.argmax(weighted_feedback)
            best_feedback_similarity = weighted_feedback[best_feedback_idx]
            
            if best_feedback_similarity > best_similarity:
                predicted_tag = self.feedback_tags[best_feedback_idx]
                best_similarity = feedback_similarities[best_feedback_idx]
        
        return predicted_tag, best_similarity
    
    def load_feedback(self):
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'rb') as f:
                    self.feedback_data = pickle.load(f)
                print(f"Loaded {len(self.feedback_data)} feedback entries.")
        except Exception as e:
            print(f"Error loading feedback: {e}")
            self.feedback_data = {}
    
    def save_feedback(self):
        try:
            with open(self.feedback_file, 'wb') as f:
                pickle.dump(self.feedback_data, f)
            print("Feedback saved successfully.")
        except Exception as e:
            print(f"Error saving feedback: {e}")
    
    def add_feedback(self, user_query, correct_tag):
        if user_query.lower() in self.feedback_data:
            self.feedback_data[user_query.lower()]['weight'] += 1
        else:
            self.feedback_data[user_query.lower()] = {
                'correct_tag': correct_tag,
                'weight': 1.5
            }
        
        self.save_feedback()
        self.update_feedback_vectors()
        print(f"Feedback recorded: '{user_query}' -> {correct_tag}")
    
    def update_feedback_vectors(self):
        if not self.feedback_data or self.vectorizer is None:
            return
        
        feedback_queries = list(self.feedback_data.keys())
        processed_queries = [self.preprocess_text(q) for q in feedback_queries]
        
        self.feedback_tags = [self.feedback_data[q]['correct_tag'] for q in feedback_queries]
        self.feedback_weights = [self.feedback_data[q]['weight'] for q in feedback_queries]
        
        self.feedback_vectors = self.vectorizer.transform(processed_queries)
    
    def save_model(self, intents_mtime):
        try:
            data_to_save = {
                'vectorizer': self.vectorizer,
                'pattern_vectors': self.pattern_vectors,
                'pattern_tags': self.pattern_tags,
                'intents_mtime': intents_mtime,
                'tag_to_intent': self.tag_to_intent
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Model saved to {self.model_file}.")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        try:
            with open(self.model_file, 'rb') as f:
                data = pickle.load(f)
            
            self.vectorizer = data['vectorizer']
            self.pattern_vectors = data['pattern_vectors']
            self.pattern_tags = data['pattern_tags']
            self.tag_to_intent = data['tag_to_intent']
            return data['intents_mtime']
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None