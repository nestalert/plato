import json
import random
import re
import os
import pickle
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
warnings.filterwarnings("ignore", category=UserWarning) 

class IntentChatbot:
    def __init__(self, class_suggestions, intents_file='intents.json', links_file='links.json', 
                 professors_file='professors.json', model_file='chatbot_model.pkl',
                 feedback_file='feedback_data.pkl'):
        self.intents_file = intents_file
        self.links_file = links_file
        self.professors_file = professors_file
        self.class_suggestions = class_suggestions
        self.model_file = model_file
        self.feedback_file = feedback_file
        self.intents = None
        self.links = None
        self.professors = None
        self.vectorizer = None
        self.pattern_vectors = None
        self.pattern_tags = []
        self.tag_to_intent = {}
        
        # Relevance feedback storage
        self.feedback_data = {}  # {user_query: {'correct_tag': tag, 'weight': weight}}
        self.feedback_vectors = None
        self.feedback_tags = []
        self.feedback_weights = []
        
        self.load_json()
        self.load_feedback()
        self.load_or_train_model()
        
    def load_json(self):
        with open(self.intents_file, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)
        with open(self.links_file, 'r', encoding='utf-8') as f:
            self.links = json.load(f)
        with open(self.professors_file, 'r', encoding='utf-8') as f:
            self.professors = json.load(f)  

    def load_feedback(self):
        """Load previously saved feedback data"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'rb') as f:
                    self.feedback_data = pickle.load(f)
                print(f"Loaded {len(self.feedback_data)} feedback entries.")
        except Exception as e:
            print(f"Error loading feedback: {e}")
            self.feedback_data = {}
    
    def save_feedback(self):
        """Save feedback data to file"""
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

    def detect_professor(self, user_input):
        user_input_lower = user_input.lower()
        
        for professor in self.professors:
            if professor['name'].lower() in user_input_lower:
                return professor
            
            last_name = professor['name'].split()[-1].lower()
            if last_name in user_input_lower:
                return professor
            
            if 'aliases' in professor:
                for alias in professor['aliases']:
                    if alias.lower() in user_input_lower:
                        return professor
        
        return None
    
    def format_professor_info(self, professor):
        response = f"Here's the contact information for {professor['name']}:\n"
        if 'title' in professor:
            response += f"Title: {professor['title']}\n"
        if 'email' in professor:
            response += f"Email: {professor['email']}\n"
        if 'phone' in professor:
            response += f"Phone: {professor['phone']}\n"
        if 'office' in professor:
            response += f"Office: {professor['office']}\n"
        if 'office_hours' in professor:
            response += f"Office Hours: {professor['office_hours']}\n"
        if 'department' in professor:
            response += f"Department: {professor['department']}\n"
        
        return response.strip()

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()

        if len(words) <= 2:  # if the query is short.
            return " ".join(words)

        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words if w not in stop_words]
        return " ".join(words)
    
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
        print("Training vector space model...")
        patterns, tags = self.prepare_training_data()
        
        # Create TF-IDF vectorizer and fit on all patterns
        self.vectorizer = TfidfVectorizer()
        self.pattern_vectors = self.vectorizer.fit_transform(patterns)
        self.pattern_tags = tags
        
        # Update feedback vectors with new vectorizer
        self.update_feedback_vectors()
        
        print("Training complete.")
        self.save_model()
    
    def save_model(self):
        try:
            intents_mtime = os.path.getmtime(self.intents_file)
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

    def load_or_train_model(self):
        intents_mtime_saved = self.load_model()
        
        if intents_mtime_saved is not None:
            current_intents_mtime = os.path.getmtime(self.intents_file)
            
            if current_intents_mtime == intents_mtime_saved:
                print(f"Loaded model from {self.model_file}. Intents file hasn't changed.")
                self.update_feedback_vectors()
                return
            else:
                print("Intents file has been modified. Re-training model.")
                self.train()
        else:
            print("Saved model not found. Training model.")
            self.train()
    
    def predict_intent(self, user_input):
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
    
    def get_response(self, user_input, confidence_threshold=0.2, ask_feedback=True):
        professor = self.detect_professor(user_input)
        if professor:
            return self.format_professor_info(professor), None
        
        predicted_tag, confidence = self.predict_intent(user_input)
        
        word_count = len(user_input.split())
        if word_count <= 2:  # If input is short, go easy
            confidence_threshold = 0.1

        if confidence < confidence_threshold:
            predicted_tag = "fallback"
        
        intent = self.tag_to_intent[predicted_tag]
        response = random.choice(intent['responses'])

        if predicted_tag == "class_selection" and self.class_suggestions:
            printed_output = "Based on your record, we recommend the following classes:\n"
            for i, course in enumerate(self.class_suggestions, start=1):
                printed_output += f"{i}. {course}\n"
            printed_output += "You can also check the Schedule at any time: [[Schedule]]"
            response = printed_output

        return response, (predicted_tag if ask_feedback else None)

    def replace_links(self, response):
        for placeholder, link_value in self.links.items():
            response = response.replace(placeholder, link_value)
        return response
    
    def get_available_intents(self):
        intents_list = []
        for intent in self.intents:
            tag = intent['tag']
            example = intent['patterns'][0] if intent['patterns'] else 'N/A'
            intents_list.append(f"{tag}: {example}")
        return intents_list

    def handle_feedback(self, last_user_input,correct_tag = None):
        print("\nWhat should the correct intent be?")
        print("Available intents:")
        for i, intent_info in enumerate(self.get_available_intents(), 1):
            print(f"  {i}. {intent_info}")
        
        correct_tag = input("\nEnter correct intent tag: ").strip()
        
        if correct_tag in self.tag_to_intent:
            self.add_feedback(last_user_input, correct_tag)
            print("Bot: Thank you! I'll remember that.\n")
        else:
            print("Bot: Invalid intent tag. Feedback not recorded.\n")


    def chat(self):
        print("\n" + "="*50 + "\n")
        print("What can I help you with?")
        print("(Type 'feedback' after any response to correct the bot)\n")
        
        last_user_input = None
        
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            user_input_lower = user_input.lower()
                
            if user_input_lower in ['quit', 'exit', 'bye']:
                print("Bot: Goodbye!")
                break
            
            # Handle feedback command
            if user_input_lower == 'feedback' and last_user_input:
                self.handle_feedback(last_user_input)
                continue
            
            response, predicted_tag = self.get_response(user_input)
            response = self.replace_links(response)
            print(f"Bot: {response}")
            
            last_user_input = user_input
            
            if predicted_tag:
                print(f"(Detected intent: {predicted_tag} | Type 'feedback' to correct)\n")
            else:
                print()
