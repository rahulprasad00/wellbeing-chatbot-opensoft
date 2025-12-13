from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import nltk
from nltk.tokenize import word_tokenize
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from question_bank import question_bank
import random
from chatgpt import chat_with_gpt4o

# Download the required data file
nltk.download('punkt_tab')

# # Tokenize each question separately and combine the results
asked_questions = set()


# Pass questions_set
def retrieve_relevant_questions(user_query, questions_set, top_k=20):
    # print("System Prompt Applied:")
    # print(system_prompt)
    try:
        # nltk.download('punkt_tab')
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})
        

        documents = []
        for question in questions_set:
            documents.extend(word_tokenize(question))
        documents_as_docs = [Document(page_content=word) for word in documents]

        vectorstore = FAISS.from_documents(documents_as_docs, embeddings)
        vectorstore.save_local("vectorstore.db")
        # retriever = vectorstore.as_retriever()

        # Embed user query using embed_query to get a 1D embedding
        query_embedding = embeddings.embed_query(user_query)

        # Embed all questions using embed_query to get 1D embeddings for each question
        question_embeddings = embeddings.embed_documents(documents)

        # Compute similarity
        similarities = cosine_similarity(
            [query_embedding], question_embeddings)[0]

        # Select top_k questions
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_questions = [questions_set[i] for i in top_indices]

        return top_questions
    except Exception as e:
        print(f"Error in retrieve_relevant_questions: {e}")
        # Fallback to returning the first top_k questions
        return questions_set[:top_k]


def detect_contradiction(user_response, chat_history):
    """Checks if the user response contradicts any previous responses."""

    if len(chat_history) < 2:
        return ""  # Not enough context for contradiction
    try:
        system_prompt = (
            "You are a helpful assistant that detects contradictions in a conversation. "
            "Your job is to analyze the user's most recent message and determine if it contradicts any of their previous responses. "
            "Only respond with a clear follow-up question if a contradiction exists, or respond with the exact word 'None' if there is no contradiction. "
            "Do not include any explanation, reasoning, or additional text—just the follow-up question or 'None'."
        )

        user_prompt = f"""
        Conversation so far:
        {chat_history}

        New user message:
        "{user_response}"

        Analyze the user's response based on the context above. 

        If there is a contradiction with any earlier response:
        - Return ONLY a clear and concise follow-up question to clarify the contradiction.

        If there is NO contradiction:
        - Return EXACTLY 'None' (case-sensitive) and nothing else.

        Do not explain or include any other text. Just return the follow-up question or 'None'.
        """

        contradiction_question = chat_with_gpt4o(system_prompt, user_prompt)
        return "" if contradiction_question.lower() == "none" else contradiction_question
    except Exception as e:
        print(f"Error in detect_contradiction: {e}")
        return ""


def generate_follow_up(user_response):
    """Generate 1 follow-up questions based on the user's response."""
    try:
        system_prompt = (
            "You are a thoughtful assistant skilled in asking meaningful follow-up questions that deepen conversations. "
            "Your job is to generate exactly ONE insightful follow-up question based on the user's latest response. "
            "Return only the question—do not include any introductions, explanations, or additional text. "
            "The question should encourage deeper thinking and explore a different angle from the user's previous response."
        )

        user_prompt = f"""
        User's latest response:
        "{user_response}"

        Generate exactly one thoughtful follow-up question:
        - It should dig deeper or explore a new angle of the user's input.
        - Do not repeat the user's wording exactly.
        - Do not add any commentary or explanation—just return the question as plain text.
        """

        follow_ups = chat_with_gpt4o(system_prompt, user_prompt)
        return follow_ups.strip()  # Return the first follow-up question
        # return [q.strip() for q in follow_ups if q.strip()][:1]
    except Exception as e:
        print(f"Error in generate_follow_up: {e}")
        return []


def select_next_question(chat_history, question_set):
    """Selects the next most relevant question that has NOT been asked."""
    remaining_questions = [q for q in question_set if q not in asked_questions]

    if not remaining_questions:
        return None  # No more questions left

    try:

        system_prompt = (
            "You are an intelligent assistant that guides a conversation by selecting the most relevant next question from a given list. "
            "You must consider the flow of the conversation and choose the most contextually appropriate question. "
            "Return only the selected question as plain text. Do not include any explanation, commentary, or formatting."
        )
        user_prompt = f"""
Here is the conversation so far:
{chat_history}

Choose the most relevant next question from the list below:
{remaining_questions}

Instructions:
- Return ONLY the selected question.
- DO NOT include any explanation or extra text.
- Return the question as plain text only.
"""

        next_question = chat_with_gpt4o(system_prompt, user_prompt)

        if next_question in remaining_questions and next_question not in asked_questions:
            asked_questions.add(next_question)
            return next_question
        else:
            # Fallback: pick a random one to ensure progress
            fallback = random.choice(remaining_questions)
            asked_questions.add(fallback)
            return fallback
    except Exception as e:
        print(f"Error in select_next_question: {e}")
        fallback = random.choice(remaining_questions)
        asked_questions.add(fallback)
        return fallback


def chatbot_conversation(shap_values, chat_history, user_response, message_type, question_set):
    """Handles the chatbot conversation logic."""

    try:
        if message_type == "welcome":
            first_shap_value = next(iter(shap_values))
            current_shap_questions = question_bank[first_shap_value].get(
                "questions", [])
            current_question = random.choice(current_shap_questions)
            asked_questions.add(current_question)
            return current_question, "normal_question"

        if message_type == "followup_1" or message_type == "normal_question":
            print(f"User Response1: {user_response}")
            contradiction_follow_up = detect_contradiction(
                user_response, chat_history)

            if contradiction_follow_up != "" and contradiction_follow_up != None and contradiction_follow_up != "None.":
                asked_questions.add(contradiction_follow_up)
                if message_type == "normal_question":
                    return contradiction_follow_up, "followup_1"
                elif message_type == "followup_1":
                    return contradiction_follow_up, "followup_2"
            else:
                print(f"User Response2: {user_response}")
                follow_up_question = generate_follow_up(user_response)
                if message_type == "normal_question":
                    return follow_up_question, "followup_1"
                elif message_type == "followup_1":
                    return follow_up_question, "followup_2"

        next_question = select_next_question(chat_history, question_set)
        asked_questions.add(next_question)
        # chat_history.append(AIMessage(content=next_question))
        return next_question, "normal_question"
    except Exception as e:
        print(f"Error in chatbot_conversation: {e}")
        return "I'm sorry, I couldn't process your request.Please try again", "normal_question"


# Run the chatbot
# chatbot_conversation()
  # Import SystemMessage and HumanMessage


# input_data = {
#     "Promotion": "Not Satisfactory",
#     "Meetings attended": "Normal",
#     "Holidays": "Less",
#     "Working Hours": "Very High",
#     "Happiness": "Unhappy"
# }

# # Convert input dict to readable string
# input_str = ", ".join([f"{k}: {v}" for k, v in input_data.items()])

# # Compose messages
# messages = [
#     SystemMessage(
#         content="You are an HR assistant AI. Given an employee's data (like promotion, holidays, work hours, mood), generate a short and professional summary (2-3 sentences) that reflects their work experience, satisfaction, and potential concerns. This gives a description with a focus on the problems of the employee"
#     ),
#     HumanMessage(
#         content=f"Employee details: {input_str}"
#     )
# ]
def generate_user_summary(input_data):
    """Generates a summary based on the user's input data."""
    # Convert input dict to readable string
    input_str = ", ".join([f"{k}: {v}" for k, v in input_data.items()])

    # Compose messages

    system_prompt = "You are an HR assistant AI. Given an employee's data (like promotion, holidays, work hours, mood), generate a short and professional summary (2-3 sentences) that reflects their work experience, satisfaction, and potential concerns. This gives a description with a focus on the problems of the employee"
    user_prompt = f"Employee details: {input_str}"
    summary = chat_with_gpt4o(system_prompt, user_prompt)
    return summary
