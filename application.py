from flask import Flask, jsonify, request
from maq102 import *
import asyncio
from flask import make_response
import re
from flask_cors import CORS
import asyncio
from dotenv import load_dotenv
from maqx104 import *
# initialize the flask app
application =  Flask(__name__)

CORS(application, resources={r"/*": {"origins": "*"}}, supports_credentials=True)


# load_dotenv()
# First we want to initialize the text conversion class
@application.before_request
def log_request_info():
    print("Headers:", request.headers)
    print("Body:", request.data)



@application.route("/", methods=["GET", "POST", "OPTIONS"])
def home():
    return "Welcome to my Flask application"


@application.route("/prompts/make_prompt", methods=["POST", "OPTIONS", "GET"])
def make_prompt():
    if request.method == "OPTIONS":
        return jsonify({"message": ""}), 200
    elif request.method == "POST":
        data = request.get_json()
        prompt = data.get("prompt_command")
        user_id = data.get("user_id")
        user_token = data.get("user_token")
        email = data.get("user_email")
        
        try:
            # Process the prompt
            prompt_check = process_prompt(prompt)
            clean_prompt = asyncio.run(prompt_check.extract_keywords())
            
            # Log clean_prompt
            # print(f"Extracted Keywords: {clean_prompt}")

            # Handle empty prompt
            if not clean_prompt:
                return jsonify({"error": "No keywords extracted", "status": 400}), 400
            
            # Get LinkedIn Contacts
            # contact_details = Contact_Details(clean_prompt, prompt)
            # linkedInContacts = asyncio.run(contact_details.fetch_contacts_from_linkedin("+234"))
            # print("contacts from linkedin are: ", linkedInContacts)
            # Save the contents in linkedInContacts to a file
            # with open("store_text.txt", "w") as file:
            #     for contact in linkedInContacts:
            #         file.write(str(contact) + "\n")
            


            # Ensure the data is in the correct format
            # if isinstance(linkedInContacts, list):
            #     linkedInContacts = [str(contact) for contact in linkedInContacts]

            # Extract information using AI
            try:
                m4 = maq4(clean_prompt)
                final_response = asyncio.run(m4.start_maq4())
                print(f"bland response: {final_response}")
   
                bland_response =  final_response.to_dict()
                # Convert CrewOutput to a dictionary if possible
               
                # if hasattr(final_response, "to_dict"):
                #     final_response = final_response.to_dict()
                # else:
                #     # Manually extract relevant information
                #     final_response = {
                #         "result": str(final_response)
                #     }
                
                # Ensure the result is JSON serializable
                
                return jsonify({"message": bland_response, "status": 200})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    
@application.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response





if __name__=="__main__":
    application.run(debug=True, port=5000)
