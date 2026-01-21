import json
import random
import os
from appointments import DatabaseManager,AppointmentHandler
from professor import ProfessorHandler
from nlp_engine import NLPProcessor

class IntentChatbot:
    def __init__(self, uname, pwd, class_suggestions, assignments,
                 intents_file='intents.json', links_file='links.json', professors_file='professors.json',
                 model_file='chatbot_model.pkl', feedback_file='feedback_data.pkl'):
        
        self.intents_file = intents_file
        self.links_file = links_file
        self.class_suggestions = class_suggestions
        self.assignments = assignments
        self.uname = uname
        self.pwd = pwd
        
        self.intents = None
        self.links = None
        self.load_json()
        
        self.db_manager = DatabaseManager()
        self.professor_handler = ProfessorHandler(professors_file)
        self.nlp_processor = NLPProcessor(self.intents, model_file, feedback_file)
        self.appointment_handler = AppointmentHandler(self.db_manager, uname)
        
        self.load_or_train_model()
        
    def load_json(self):
        with open(self.intents_file, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)
                
        with open(self.links_file, 'r', encoding='utf-8') as f:
            self.links = json.load(f)

    def load_or_train_model(self):
        intents_mtime_saved = self.nlp_processor.load_model()
        
        if intents_mtime_saved is not None:
            current_intents_mtime = os.path.getmtime(self.intents_file)
            
            if current_intents_mtime == intents_mtime_saved:
                print(f"Loaded model from {self.nlp_processor.model_file}. Intents file hasn't changed.")
                self.nlp_processor.load_feedback()
                self.nlp_processor.update_feedback_vectors()
                return
            else:
                print("Intents file has been modified. Re-training model.")
                self.nlp_processor.train()
                self.nlp_processor.save_model(current_intents_mtime)
        else:
            print("Saved model not found. Training model.")
            self.nlp_processor.train()
            current_intents_mtime = os.path.getmtime(self.intents_file)
            self.nlp_processor.save_model(current_intents_mtime)
        
        self.nlp_processor.load_feedback()
    
    def get_response(self, user_input, confidence_threshold=0.2, ask_feedback=True):
        if self.appointment_handler.is_active():
            response = self.appointment_handler.handle_flow(user_input)
            return response, "appointment_flow"
        
        if user_input.lower() == "create appointment":
            return self.appointment_handler.start_create(), "appointment_flow"

        if user_input.lower() == "delete appointment":
            return self.appointment_handler.start_delete(), "appointment_flow"

        professor = self.professor_handler.detect_professor(user_input)
        if professor:
            return self.professor_handler.format_professor_info(professor), None
        
        predicted_tag, confidence = self.nlp_processor.predict_intent(user_input)
        
        word_count = len(user_input.split())
        if word_count <= 3:
            confidence_threshold = 0.1

        if confidence < confidence_threshold:
            predicted_tag = "fallback"
        
        intent = self.nlp_processor.tag_to_intent[predicted_tag]
        response = random.choice(intent['responses'])

        if predicted_tag == "class_selection" and self.class_suggestions:
            response = self._format_class_suggestions()
        
        if predicted_tag == "assignment":
            response = self._format_assignments()

        if predicted_tag == "appointment":
            response = self._format_appointments()

        return response, (predicted_tag if ask_feedback else None)

    def _format_class_suggestions(self):
        output = "Based on your record, we recommend the following classes:\n"
        for i, course in enumerate(self.class_suggestions, start=1):
            output += f"{i}. {course}\n"
        output += "You can also check the Schedule at any time: [[Schedule]]"
        return output
    
    def _format_assignments(self):
        output = "Checking your assignments now...\n"
        if not self.assignments:
            output += "You have no assignments. Good for you!"
        else:
            for i, assignment in enumerate(self.assignments, start=1):
                output += (
                    f"{i}. {assignment['due_status']} ({assignment['end_date']}): "
                    f"{assignment['name']}. {assignment['description']}. IN: "
                    f"{assignment['course_name']}\n"
                )
        return output

    def _format_appointments(self):
        output = self.appointment_handler.read_appointments()
        output += "\nIf you wish to add an appointment, type 'create appointment'."
        output += "\nIf you wish to delete an appointment, type 'delete appointment'."
        return output

    def replace_links(self, response):
        for placeholder, link_value in self.links.items():
            response = response.replace(placeholder, link_value)
        return response
    
    def get_available_intents(self):
        intents_list = []
        for intent in self.intents:
            tag = intent['tag']
            if tag == "fallback":
                continue
            example = intent['patterns'][0] if intent['patterns'] else 'N/A'
            intents_list.append(f"{tag}: {example}")
        return intents_list

    def handle_feedback(self, last_user_input, correct_tag=None):
        print("\nWhat should the correct intent be?")
        print("Available intents:")
        for i, intent_info in enumerate(self.get_available_intents(), 1):
            print(f"  {i}. {intent_info}")
        
        correct_tag = input("\nEnter correct intent tag: ").strip()
        
        if correct_tag in self.nlp_processor.tag_to_intent:
            self.nlp_processor.add_feedback(last_user_input, correct_tag)
            print("Bot: Thank you! I'll remember that.\n")
        else:
            print("Bot: Invalid intent tag. Feedback not recorded.\n")