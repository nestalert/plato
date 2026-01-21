import json

class ProfessorHandler:
    
    def __init__(self, professors_file='professors.json'):
        self.professors_file = professors_file
        self.professors = None
        self.load_professors()
    
    def load_professors(self):
        with open(self.professors_file, 'r', encoding='utf-8') as f:
            self.professors = json.load(f)
    
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