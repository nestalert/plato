from bot import IntentChatbot
from counsel import suggestions

from assignments import check_assignments
def main():
    uname="ice2"
    pwd="11051989"
    result = suggestions(uname, pwd)
    class_suggestions,welcome_message = result
    assignments = check_assignments(uname,pwd)
    bot = IntentChatbot(
            uname=uname, 
            pwd=pwd, 
            class_suggestions=class_suggestions, 
            assignments=assignments
        )

    print(welcome_message)
    print("\nChatbot is ready! (Type 'exit' or 'quit' to stop)")
    print("-" * 30)

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("Bot: Goodbye!")
            break

        if not user_input:
            continue

        response, intent_tag = bot.get_response(user_input)

        final_text = bot.replace_links(response)
        print(f"Bot: {final_text}")

if __name__ == "__main__":
    main()