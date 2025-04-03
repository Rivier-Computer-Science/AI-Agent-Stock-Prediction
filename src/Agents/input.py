import json
import os

from openai import OpenAI

# Fetch the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("OpenAI API key is not set in environment variables!")

# Initialize OpenAI Client
client = OpenAI(api_key="openai_api_key")

# API key for OpenAI
client.api_key = openai_api_key

# Function to interact with ChatGPT and extract details from the query
def use_openai(prompt):
    try:
        # Get a response from ChatGPT using the prompt
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error interacting with OpenAI: {e}")
        return None

# Function to process user input and return the extracted data in JSON format
def process_user_input(user_query):
    # user_query = input("Enter your market prediction query: ")

    # Prepare the prompt for ChatGPT to extract stock details
    prompt = f"""
    Please extract the following information from the user's query:
    - Stock symbol (e.g., 'AAPL', 'GOOG', etc.)
    - Percentage change (e.g., '5%', '-3%', etc.)
    - Time frame (e.g., 'next year', '6 months', 'end of this month', etc.)

    User Query: {user_query}

    Provide the extracted information as a JSON object. If any of the information is missing, do not include it in the JSON.
    """

    # Get the response from OpenAI
    response = use_openai(prompt)

    if response:
        try:
            # Try to parse the response as JSON
            extracted_data = json.loads(response)
            print("Extracted Data (JSON format):")
            print(json.dumps(extracted_data, indent=4))

            # Save the extracted data to a JSON file for further use in getting the stock data
            with open('extracted_data.json', 'w') as outfile:
                json.dump(extracted_data, outfile, indent=4)

            print("\nExtracted data has been saved to 'extracted_data.json'.")
            return extracted_data  # Return the extracted data for use in getting the stock data
        except json.JSONDecodeError:
            print(f"Error decoding JSON response: {response}")
    else:
        print("Failed to extract relevant details.")
        return None