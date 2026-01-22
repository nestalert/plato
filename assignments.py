import mysql.connector
from datetime import datetime
from db_connector import db_manager

def check_assignments(uname, pwd):
    try:
        mydb = db_manager.get_connection()
        mycursor = mydb.cursor()
        
    except mysql.connector.Error as err:
        print("Server is offline, logging in as guest...")
        return [] 

    mycursor = mydb.cursor()

    while True:
        if uname is None:
            uname = "guest"
            pwd = "guest"
            
        if uname.lower() in ("guest"):
            mycursor.close()
            mydb.close()
            return []
            
        mycursor.execute("SELECT * FROM STUDENT WHERE STUDENT_ID = %s AND PASSWORD = %s", (uname, pwd))

        student = mycursor.fetchone()

        if student:
            break  
        else:
            print("Invalid username or password. Please try again.")
            uname = input("Username: ")
            pwd = input("Password: ")

    student_id = student[0]

    query = """
        SELECT a.END_DATE, a.NAME, a.DESCRIPTION, c.COURSE, a.CLASS_ID
        FROM ASSIGNMENT a
        JOIN CLASS c ON a.CLASS_ID = c.CLASS_ID
        WHERE a.STUDENT_ID = %s AND a.SUBMITTED = FALSE
        ORDER BY a.END_DATE ASC
    """

    assignments_list = [] 

    try:
        mycursor.execute(query, (student_id,))
        assignments = mycursor.fetchall()

        if not assignments:
            print("\nYou currently have no pending assignments.")
        else:
            
            now = datetime.now()

            for asm in assignments:
                end_date_dt = asm[0]
                asm_name = asm[1]
                description = asm[2]
                course = asm[3]
                class_id = asm[4]
                time_difference = end_date_dt - now
                
                days_until_due = time_difference.days
                if time_difference.total_seconds() > 0:
                    pass

                due_string = ""
                total_seconds_diff = time_difference.total_seconds()
                
                if total_seconds_diff < 0:
                    due_string = "OVERDUE"
                elif days_until_due <= 0:
                    due_string = "Due soon"
                else:
                    days_until_due = max(0, days_until_due + 1) 
                    due_string = f"Due in {days_until_due} days"


                formatted_date = end_date_dt.strftime('%d %B %Y, %H:%M:%S')

                assignment_obj = {
                    "due_status": due_string,
                    "end_date": formatted_date,
                    "name": asm_name,
                    "description": description,
                    "course_name": course,
                    "class_id": class_id
                }
                
                assignments_list.append(assignment_obj)
            
            return assignments_list

    except mysql.connector.Error as err:
        print(f"There was an error fetching your assignments: {err}")
        return []

    finally:
        mycursor.close()
        mydb.close()