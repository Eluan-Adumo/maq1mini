from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from apify_client import ApifyClientAsync
from apify_client import ApifyClient
from pydantic import BaseModel
import asyncio
import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool
from typing import List
import ast
import re
import os
from apify_client.clients.resource_clients import TaskClientAsync
from dotenv import load_dotenv
from tools import *
from urllib.parse import urljoin
import difflib 
import json
import litellm 
from litellm import completion
from litellm import LiteLLM
from llm_integration import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List



llm = LLM(model="gpt-4o-mini", temperature=0)
load_dotenv()




class ContactDetailModel(BaseModel):
    name: str
    phone_number: str
    profile_image: str
    email_address: str
    company_name : str
    company_logo: str
    contact_position: str
    ai_insight: str
    contact_linked_in: str
    company_website: str

class ContactDetailsModel(BaseModel):
    contacts: list[ContactDetailModel]


class ExecutiveModel(BaseModel):
    exec_name: str
    exec_phone_number: str
    exec_email_address: str
    exec_position: str
    exec_linkedin: str = ""
    exec_profile_image: str = ""

class CompanyModel(BaseModel):
    company_name: str
    company_website: str
    company_logo: str = ""
    ai_summary: str
    company_execs: List[ExecutiveModel]

class CompaniesModel(BaseModel):
    companies: List[CompanyModel]

class maq4:

    def __init__(self, prompt):
        self.prompt = prompt
        print("Starting maq4....")

    
    async def start_maq4(self):
        try:
            get_name_agent = Agent(
                role = f"You role is to take the information [{self.prompt}], do a google search and get companies, businesses and organizations that solve that problem. Also return the links to their leadership board. Try to return the list of your topmost results and pages with highest rankings...",
                goal = "A list of websites links and the company names and a list of their leadership board. in a python list",
                backstory = ("A resource getting agent who produces the best sources and links to any question or problem that aid research the most."),
                verbose = False,
                allow_delegation= True,
                tools = [google_search_tool_decorated],
                llm = llm
                # max_iter = 5,
                # max_rpm = 5,
                # memory= False,
                # cache = False,
            )

            get_name_task = Task(
                description=f"You have been given a prompt by the user [{self.prompt}], make a search and return the companies and their website links that best solve the problem.. ",
                expected_output = "A list of websites links and the company names in a python list.",
                agent = get_name_agent,
                inputs = {"data" : self.prompt}
            )



            get_summary_agent = Agent(
                role = f"Your job is to take the links for the websites you are provided, visit them and then to return the summary of what each company does. Make it a detailed summary between 150 characters and 300 characters.. ALSO : Your responsibility is to get the official logo of the companies in the data you have received from the other agent. Make a search specifically to get the logo and return the links you get. From the returned data, you can get the logo. parse in the website link in order to get the data back. You can also pass the link of the website into your crawl_website tool to get the crawled data. You'll see html tags. One of them will be the img tag of the logo. Find that one and return the src attribute. Note that you must also return the description from the previous data you received. Do not overwrite the data. Just append your results to what you were give. Everything should be in a python dictionary.",
                goal = "To provide a summary of each copmanies activities and also To crawl into the copmany website and return a link to thier logo appending your links as a list to the received data",
                backstory = ("You are a concise and straight forward agent who takes the most bulk of information and summarises them. You also specialize in crawling websites, understanding the source code and looking for important details to of these copmanies"),
                verbose = True,
                allow_delegation=True,
                tools = [google_search_tool_decorated, crawl_website],
                llm = llm
            )



            get_summary_task = Task(
                description = f"You have been given a list of companies and their website links. Visit those links and return a description of what the company / organization does in not more than 300 characters... Then, (A) Crawl all the websites given to you. (B) For each website, look into their source code and try to extract a link to their logo. Note. You must only return one logo per company. Prioritise the png logos over the rest...",
                expected_output="A list of company names, their links and their description. You are to call that summary agent insight. Your responses should be in a dictionary",
                agent = get_summary_agent,
                context=[get_name_task]
            )


            # get_logo_agent = Agent(
            #     role  = "Your responsibility is to get the official logo of the companies in the data you have received from the other agent. Make a search specifically to get the logo and return the links you get. From the returned data, you can get the logo. parse in the website link in order to get the data back. You can also pass the link of the website into your crawl_website tool to get the crawled data. You'll see html tags. One of them will be the img tag of the logo. Find that one and return the src attribute. Note that you must also return the description from the previous data you received. Do not overwrite the data. Just append your results to what you were give. Everything should be in a python dictionary.",
            #     goal = "To crawl into the copmany website and return a link to thier logo appending your links as a list to the received data",
            #     backstory = ("You specialize in crawling websites, understanding the source code and looking for important details to of these copmanies"),
            #     verbose = True,
            #     tools = [google_search_tool_decorated, crawl_website],
            #     llm = llm,
            #     cache = False,
            #     allow_delegation=True
            # )

            # get_logo_task = Task(
            #     description = "(A) Crawl all the websites given to you. (B) For each website, look into their source code and try to extract a link to their logo. Note. You must only return one logo per company. Prioritise the png logos over the rest...",
            #     expected_output = "Add your result as a part of the data you already received. Organizse the data into a list",
            #     agent = get_logo_agent,
            #     context = [get_name_task],

            # )



            get_key_players_agent = Agent(
                role  = "For each of the organizations in the data you have been passed, Do a search and get the key players, executives and decision makers of the companies in the list of companies you have received. Return their names. I need to see names of CEOS, HR managers, CFOs, CTOs, etc. Append your results to the results you got...Avoid placeholder names like John Doe, Jane Smith, Jane Doe, etc. Try to get the real names or just leave it blank. Also get the most recent information. Do not return execs who have changed roles, have recently been dropped or are not actively serving. ",
                goal = "To research and find out the decision makers and leaders of the companies given to you. Produce a structured python dictionary object from your research.",
                backstory = ("You specialize in fishing out the current executives of companies."),
                verbose = False,
                tools = [google_search_tool_decorated],
                llm = llm,
                allow_delegation=True
            )


            get_key_players_task = Task(
                description = "Make a search for the key players and decision makers for every company in the data you have gotten. Do not overwrite any of the previous data, just add yours to the output..",
                expected_output = "Key players and executive board of the organizations given. Add your result as a part of the data you already received. Organizse the data into a Python dictionary.",
                agent = get_key_players_agent,
                inputs = {"data" : self.prompt},
                context = [get_summary_task]
            )


            get_linkedin_profiles_agent = Agent(
                role="Use the data about key players you have received and perform a Google search to extract LinkedIn profile URLs for those individuals. Return the LinkedIn profile links organized by name. Organize your response into a dictionary of two items. First item in the dictionary should be the previous items in the data you have received. The next item in the dictionary should include a list of all the linkedin profiles of the key players as you have received. This linkedin profile links should also be in the previous item in the dictionary, but  I need a list of all of them in the a separate list... ",
                goal="To retrieve LinkedIn profile URLs of the executives provided and place them in a separate list as a value in the previous dictionary",
                backstory=("An agent focusing on locating professional profiles"),
                verbose=False,
                tools=[google_search_tool_decorated],
                llm = llm,
                allow_delegation=True
            )

            get_linkedin_profiles_task = Task(
                description="Take the data you have received from the other agent. Make a search to get the individual linkedin profiles links",
                expected_output="Add LinkedIn profile URLs for each key player to the data. Organize the data in a dictionary... ",
                agent=get_linkedin_profiles_agent,
                inputs={"data": self.prompt},
                context=[get_key_players_task]
            )



            get_key_players_contact_agent = Agent(
                role  = "Take the data you have received, extract the linkedin urls and place them into a list. Take this string and send it into your tool. Get the profile data from the tool. Organize it and make it readable and easy to understand. take away unneccessary characters. Also remove things like posts, work experiences and other things. Put all the data into one big string. Each profile data separated from the next...",
                goal = "To make the profile information of the users readable and understandable and organized",
                backstory = ("You specialize in fishing Contact information from a huge data set."),
                verbose = True,
                tools = [apify_get_profile_details_tool],
                llm = llm,
                allow_delegation=True
            )


            get_key_players_contact_task = Task(
                description = "To get profile data using your tool from the linkedin profiles of the executives of the organizations in given to you.",
                expected_output = "Add your result as a part of the data you already received. Organizse the data into a Dictionary.",
                agent = get_key_players_contact_agent,
                context = [get_linkedin_profiles_task],
                
            )

            # get_key_players_details_refined_agent = Agent(
            #     role = "Take the data you have received and pass it to your tool.  Your tool is to help you get out the user names, user profile photo, their phone number and their email addresses. Take These sorted data you get and append it to previous one you have. Replacing information like user phone number with the new one you have gotten.",
            #     goal = "To get a concise clear detailed contact information of the key players given to you",
            #     backstory = ("You know how to filter through records and analyse them bringing out the best most desired result from them."),
            #     verbose = False,
            #     tools = [extract_user_profile],
            #     lm = llm,
            #     allow_delegation=False
            # )

            get_key_players_details_refined_agent = Agent(
                role = "Take the data you have received and try to extract contact information per person. Extract their profile photos, user names, phone numbers, email addresses and any other relevant contact information given in the dataset.",
                goal = "To get a concise clear detailed contact information of the key players given to you",
                backstory = ("You know how to filter through records and analyse them bringing out the best most desired result from them."),
                verbose = False,
                tools = [],
                llm = llm,
                allow_delegation=False
            )


            get_key_players_details_refined_task = Task(
                description = "filter through the massive datasets to get out the relevant important information. Namely the contact details (phone numbers, email address) and profile picture of each user.",
                expected_output="A dictionary of the key players with their contact details: Name, profile_photo, linkedin url, phone number, email address. Append your result to your already received data.",
                agent = get_key_players_details_refined_agent,
                input = {"data": self.prompt},
                context = [get_key_players_contact_task],
                output_json=CompaniesModel

            )


            maq3Crew = Crew(
                agents= [get_name_agent,
                        get_summary_agent, 
                        get_key_players_agent, 
                        get_linkedin_profiles_agent, 
                        get_key_players_contact_agent,
                        get_key_players_details_refined_agent],
                
                tasks = [get_name_task, 
                        get_summary_task, 
                        get_key_players_task, 
                        get_linkedin_profiles_task, 
                        get_key_players_contact_task,
                        get_key_players_details_refined_task],
                verbose = True,
                process=Process.sequential,
            )
        
        except Exception as e:
            print(f"An error in the crew occured: {str(e)}")
            return str(e)
        
        try:
            result = maq3Crew.kickoff(inputs={"question" : self.prompt})
            return result
        except Exception as e:
            print(f"An error occured: {str(e)}")
            return str(e)
    


# if __name__ == "__main__":
#     maq4 = maq4("Real Esate developers in NIgeria")
#     response = asyncio.run(maq4.start_maq4())
#     print(response)