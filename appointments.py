import mysql.connector
from datetime import datetime
    
class DatabaseManager:
    
    def __init__(self, host="localhost", user="root", password="", database="uniwa"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
    
    def _get_connection(self):
        return mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
    
    def save_appointment(self, student_id, name, description, app_time):
        try:
            mydb = self._get_connection()
            cursor = mydb.cursor()
            
            query = """INSERT INTO APPOINTMENT 
                    (STUDENT_ID, NAME, DESCRIPTION, APP_TIME) 
                    VALUES (%s, %s, %s, %s)"""
            values = (student_id, name, description, app_time)
            
            cursor.execute(query, values)
            mydb.commit()
            cursor.close()
            mydb.close()
            return True, "Appointment successfully scheduled!"
        except Exception as e:
            print(f"DB Error: {e}")
            return False, "Failed to write appointment to database."
    
    def delete_appointment(self, appointment_id, student_id):
        try:
            mydb = self._get_connection()
            cursor = mydb.cursor()
            
            check_query = "SELECT * FROM APPOINTMENT WHERE APPOINTMENT_ID = %s AND STUDENT_ID = %s"
            cursor.execute(check_query, (appointment_id, student_id))
            if not cursor.fetchone():
                cursor.close()
                mydb.close()
                return False, "Appointment not found or you don't have permission to delete it."

            delete_query = "DELETE FROM APPOINTMENT WHERE APPOINTMENT_ID = %s"
            cursor.execute(delete_query, (appointment_id,))
            mydb.commit()
            cursor.close()
            mydb.close()
            return True, f"Appointment {appointment_id} has been deleted."
        except Exception as e:
            print(f"DB Error: {e}")
            return False, "Failed to delete appointment."
    
    def get_appointments(self, student_id):
        try:
            mydb = self._get_connection()
        except mysql.connector.Error as err:
            print("Server is offline, logging in as guest...")
            return []

        mycursor = mydb.cursor()
        query = """
            SELECT APPOINTMENT_ID, NAME, DESCRIPTION, APP_TIME
            FROM APPOINTMENT
            WHERE STUDENT_ID = %s
        """
        mycursor.execute(query, (student_id,))
        results = mycursor.fetchall()
        mycursor.close()
        mydb.close()
        
        return results
    
    def format_appointments(self, appointments):
        if not appointments:
            return "No appointments were found."
        
        now = datetime.now()
        formatted = ["=========="]
        
        for appointment_id, name, description, app_time in appointments:
            time_difference = app_time - now
            days_until = time_difference.days
            total_seconds_diff = time_difference.total_seconds()
            
            if total_seconds_diff < 0:
                continue
            else:
                formatted.append(
                    f"{name}\n In {days_until} days ({app_time})\n {description}\n=========="
                )
        
        return "\n".join(formatted)


class AppointmentHandler:
    
    def __init__(self, db_manager, student_id):
        self.db_manager = db_manager
        self.student_id = student_id
        self.context = None
        self.buffer = {}
    
    def reset(self):
        self.context = None
        self.buffer = {}
    
    def is_active(self):
        return self.context is not None
    
    def handle_flow(self, user_input):
        user_input = user_input.strip()
        
        if user_input.lower() in ['cancel', 'stop', 'quit']:
            self.reset()
            return "Operation cancelled."

        if self.context == "appt_create_name":
            self.buffer['name'] = user_input
            self.context = "appt_create_desc"
            return "Type a short description about the appointment."

        elif self.context == "appt_create_desc":
            self.buffer['description'] = user_input
            self.context = "appt_create_time"
            return "Give the date and time of the appointment (e.g., '2023-12-01 14:00')"

        elif self.context == "appt_create_time":
            try:
                valid_date = datetime.strptime(user_input, "%Y-%m-%d %H:%M")
                self.buffer['app_time'] = user_input 
                
                success, msg = self.db_manager.save_appointment(
                    self.student_id,
                    self.buffer['name'],
                    self.buffer['description'],
                    self.buffer['app_time']
                )
                
                self.reset()
                return msg
                
            except ValueError:
                return "Invalid format. Please use YYYY-MM-DD HH:MM (e.g., 2023-12-01 14:00)."

        elif self.context == "appt_delete_select":
            if not user_input.isdigit():
                return "Please enter a valid numeric ID."
                
            appt_id = int(user_input)
            success, msg = self.db_manager.delete_appointment(appt_id, self.student_id)
            
            self.reset()
            return msg
        
        else:
            return "Error: Unknown state."
    
    def start_create(self):
        self.context = "appt_create_name"
        return "What is the title of the appointment?"
    
    def start_delete(self):
        appointments = self.db_manager.get_appointments(self.student_id)
        formatted = self.db_manager.format_appointments(appointments)
        
        if "No appointments were found." in formatted:
            self.reset()
            return "You have no appointments to delete."
        else:
            self.context = "appt_delete_select"
            return f"Here are your current appointments:\n{formatted}\n\nPlease enter the ID of the appointment you wish to delete."
    
    def read_appointments(self):
        appointments = self.db_manager.get_appointments(self.student_id)
        formatted = self.db_manager.format_appointments(appointments)
        return formatted