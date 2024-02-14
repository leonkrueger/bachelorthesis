ASK_USER_FOR_FEEDBACK = False


def yes_or_no_question(question: str, default_answer: bool = False) -> bool:
    # Ask the user for feedback, that can be answered with 'yes' or 'no' question
    # Returns whether the answer was 'yes'
    if not ASK_USER_FOR_FEEDBACK:
        return default_answer

    answer = ""
    print(question)
    while True:
        print("Yes / No:", end=" ")
        answer = input().strip().lower()

        if answer == "y" or answer == "yes":
            return True
        if answer == "n" or answer == "no":
            return False
