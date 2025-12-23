from tfidf import get_tfidf_matrix, get_3_best_answers

tfidf, sparse_matrix = get_tfidf_matrix()

while True:
    user_input = input("Ask a question or type 'quit' to exit\n").lower()
    if user_input == "quit":
        break
    most_relevant_FAQs = get_3_best_answers(tfidf, sparse_matrix, user_input)
    if most_relevant_FAQs[0]['score'] == 0:
        print("Did not find a matching question!")
    else:
        print("Best matching answer:")
        print(most_relevant_FAQs[0]['answer'])
        print("\nTop 3 most relevant FAQs:")
        for i, obj in enumerate(most_relevant_FAQs):
            print(f"{i+1}. {obj['question']}")
            print(f"Answer: {obj['answer']}")
            print(f"Score: {obj['score']}\n")
    