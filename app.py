from flask import Flask, request, jsonify, render_template
from counsel import suggestions 
from bot import IntentChatbot
from assignments import check_assignments
GLOBAL_BOT = None 
WELCOME_MESSAGE = None
BOT_INITIALIZED = False

def initialize_chatbot(uname, pwd):
    global GLOBAL_BOT
    global WELCOME_MESSAGE
    global BOT_INITIALIZED

    try:
        result = suggestions(uname, pwd)
        if result is None:
            class_suggestions = None
            welcome_message = "Logging in as guest..."
            assignments = None
        else:
            class_suggestions,welcome_message = result
            assignments = check_assignments(uname,pwd)

        GLOBAL_BOT = IntentChatbot(uname,pwd,class_suggestions,assignments)
        WELCOME_MESSAGE = welcome_message
        BOT_INITIALIZED = True
        return True, welcome_message
    except Exception as e:
        print(f"ERROR: Failed to initialize chatbot. Details: {e}")
        return False, str(e)


app = Flask(__name__)

@app.route('/init_bot', methods=['GET'])
def init_bot():
    uname = request.args.get('uname')
    pwd = request.args.get('pwd')
    
    if not uname or not pwd:
        return jsonify({
            "message": "Error: Both 'uname' and 'pwd' query parameters are required for initialization."
        }), 400
        
    success, result_message = initialize_chatbot(uname, pwd)
    
    if success:
        return jsonify({
            "status": "success",
            "message": "Chatbot initialized.",
            "welcome_message": WELCOME_MESSAGE
        })
    else:
        return jsonify({
            "status": "error",
            "message": f"Initialization failed: {result_message}"
        }), 500

def get_bot_response(user_input):
    if GLOBAL_BOT is None:
        return "I'm sorry, the chatbot service is currently unavailable."
        
    user_input = user_input.lower().strip()
        
    response, predicted_tag = GLOBAL_BOT.get_response(user_input)
    response = GLOBAL_BOT.replace_links(response)
    
    return response, predicted_tag

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    if GLOBAL_BOT is None:
         return jsonify({
         }), 503 
         
    if not request.is_json:
        return jsonify({"response": "Error: Request must be JSON."}), 400
        
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({"response": "Please enter a message."})
        
    bot_response,predicted_tag = get_bot_response(user_message)
    
    return jsonify({'response': bot_response,'predicted_tag': predicted_tag})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    if not data or 'last_user_input' not in data or 'correct_intent' not in data:
        return jsonify({"error": "Missing 'last_user_input' or 'correct_intent' in request"}), 400

    last_user_input = data['last_user_input']
    correct_intent = data['correct_intent']

    try:
        GLOBAL_BOT.add_feedback(last_user_input, correct_intent)
        return jsonify({"message": "Thank you! I'll remember that."}), 200
    except KeyError:
        return jsonify({"error": "Invalid intent tag. Feedback not recorded."}), 400

@app.route('/get_intents', methods=['GET'])
def get_intents():
    try:
        intents = GLOBAL_BOT.get_available_intents()
        return jsonify({"intents": intents}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve intents: {str(e)}"}), 500
   
if __name__ == '__main__':
    app.run()
