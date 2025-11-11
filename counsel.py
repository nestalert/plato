import mysql.connector

def suggestions(uname, pwd, period="F"):
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="uniwa"
        )
    except mysql.connector.Error as err:
        print("Server is offline, logging in as guest...")
        return

    mycursor = mydb.cursor()

    while True:
        if uname == None:
            uname = "guest"
        if uname.lower() in ("guest") :
            print("Logging in as guest...")
            return
        mycursor.execute("SELECT * FROM STUDENT WHERE STUDENT_ID = %s AND PASSWORD = %s",(uname, pwd))

        student = mycursor.fetchone()

        if student:
            break
        else:
            print("Invalid username or password. Please try again.")
            uname = input("Username: ")
            pwd = input("Password: ")

    name = student[2]
    semester_student = student[4]

    mycursor.execute("SELECT * FROM CLASS WHERE SEMESTER >= %s AND PERIOD = %s", (semester_student, period))
    eligible = mycursor.fetchall()

    mycursor.execute("SELECT * FROM GRADE WHERE STUDENT_ID = %s AND PASSED = FALSE", (uname,))
    pending = mycursor.fetchall()

    mycursor.execute("SELECT C.ORIENTATION FROM CLASS AS C WHERE C.CLASS_ID IN (SELECT G.CLASS_ID FROM GRADE AS G WHERE G.PASSED = TRUE AND G.STUDENT_ID = %s)", (uname,))
    orientation = mycursor.fetchall()

    class_details = {}
    elective = "M"
    mandatory = "M"
    elective_S = 0
    elective_H = 0
    elective_N = 0
    for row in orientation:
        type = row[0]
        if type == "ES":
            elective_S += 1
        if type == "EH":
            elective_H += 1
        if type == "EN":
            elective_N += 1
    if(elective_S>=elective_H and elective_S>=elective_N):
        elective = "ES"
        mandatory = "MS"
    if(elective_H>=elective_S and elective_H>=elective_N):
        elective = "EH"
        mandatory = "MH"
    if(elective_N>=elective_H and elective_N>=elective_S):
        elective = "EN"
        mandatory = "MN"

    for row in eligible:
        class_id = row[1]
        course_name = row[0]
        semester = row[2]
        orientation = row[7]
        class_details[class_id] = (course_name, semester, orientation)

    suggestable_classes = []
    for row in pending:
        class_id = row[2]
        grade = row[3]
        if class_id in class_details:
            course_name, semester, orientation = class_details[class_id]
            suggestable_classes.append({
                'course_name': course_name,
                'class_id': class_id,
                'grade': grade,
                'semester': semester,
                'orientation': orientation
            })

    def sort_key_function(class_info):
        almost_passed_priority = not (class_info['grade'] >= 3.5)
        mandatory_priority = not (class_info['orientation'] == 'M')
        mandatory_elective_priority = not (class_info['orientation'] == mandatory)
        elective_priority = not (class_info['orientation'] == elective)
        semester_priority = class_info['semester']
        return (almost_passed_priority, mandatory_priority,mandatory_elective_priority,elective_priority, semester_priority)

    suggestable_classes.sort(key=sort_key_function)

    suggestions = [c['course_name'] for c in suggestable_classes[:5]]

    period = "Fall." if period == "F" else "Spring."
    message = f"Welcome, {name}. You are on semester {semester_student}. The period is: {period}"
    
    mycursor.close()
    mydb.close()

    return suggestions, message

    