University chatbot model prototype for University of West Attica, Department of Informatics and Computer Engineering.

Python bot that can be accessed directly via testing.py, or a basic website with Flask.

### Features
- Has two models: tfidf, and transformer (all-MiniLM-L6-v2)
- Answers general Q&A, with dynamic link replacement for relevant URLs
- Can query your grades and suggest classes to take
- Reminds you of upcoming assigments
- Reads, creates and deletes appointments, your own personal calendar
- Ability to give feedback and improve the model

### Setup

To make the bot work in any mode, make sure your packages are up to date:

> pip install -r .\requirements.txt

You also need to setup the SQL database located on services.sql, and change the connection information on db_connector.py to point to your server.

### Running

To run the bot in python mode, simply run testing.py:
> py ./testing.py

To run the website, you need to initialize Flask: 
> flask run

And open your browser to localhost:5000.

If you are on Windows and run Firefox, I added a start.cmd that does it for you.
