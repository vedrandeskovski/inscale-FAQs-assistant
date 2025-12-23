# FAQs retrieval - Inscale task

The task is to create chatbot-like customer service, based on some company's FAQs.

## Dummy data

In my `data.py` I created a list of objects that represent FAQs from a company called 'Zappos'. This will be my dummy data.

## Embedding model

As an embedding model I chose a non-neural TF-IDF sparse matrix, due to its simplicity and easy lookup access. A TF-IDF score represents how much a given word is important to a given document. In this case a document is equivalent to a question from my data. I use a TFIDFVectorizer object from the `sklearn` library.

## Fitting the model

First thing I do is load all the questions in a list. Then, I fit and transform the list with the TFIDFVectorizer object. After fitting and transforming, as a result I get a sparse matrix, where each element represents a TF-IDF score, with dimensions NxM, where:

- N is the number of documents (questions)
- M is the total number of unique words (vocabulary)

A greater TF-IDF score means that the given word is unique or important to the given document related to the corpus. A low score means that the word is irrelevant to the document.

## The model in practice

The main loop of the script consists of asking for input from the user (the question that he would like to ask). When the user presses `ENTER`, these steps take place:

1. The input is taken and it is **only** transformed via the TFIDFVectorizer object. This returns a vector, whose values consist of TF-IDF scores of the user's input.
2. I take this vector, together with the sparse matrix and run them through a `cosine_similarity` function by `sklearn`. This function returns a 2D list containing the cosine similarity scores between the user's input and the individual vectors of the sparse matrix. At the end this list is flattened to a 1D list.
3. Using numpy's `argsort` method I return the indices of the 3 elements that have the highest cosine similarity score.
4. With these 3 indices I build the list of objects of the end result. Since the length of the data (FAQs) is the same as the length of the sparse matrix, I only need to perform an index lookup to get the {question, answer} object from the data.
5. The 0-th element in this list will always be the FAQ that is most similar to the user's input (based on cosine similarity).

## Tools used

python, sklearn, numpy

## How to run the project

1. navigate to the project directory via command line
2. run `python main.py`
