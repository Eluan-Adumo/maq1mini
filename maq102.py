from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from apify_client import ApifyClientAsync
from apify_client import ApifyClient
from pydantic import BaseModel
import asyncio
import requests
from bs4 import BeautifulSoup
import ast
import re
from apify_client.clients.resource_clients import TaskClientAsync
from dotenv import load_dotenv
from tools import *
import ast
import json
import os
# from phi.agent import Agent
# from phi.tools.openai import OpenAIChat
# from phi.tools.yfinance import YFinanceTools
# from phi.tools.duckduckgo import DuckDuckGo, DuckDuckGoSearchException


load_dotenv()

llm = LLM(model="gpt-4o-mini", temperature=0)

profile_names = {}

def sanitize_name(name: str) -> str:
    """
    Sanitize the task name to comply with Apify API requirements.
    Args:
        name (str): The original name.
    Returns:
        str: A sanitized name.
    """
    name = name.lower()
    name = re.sub(r'[^a-z0-9-]', '-', name)
    name = re.sub(r'-+', '-', name)
    name = name.strip('-')
    return name[:50]  # Truncate to 50 characters for safety


# END FUNCTION HERE





def fetch_plain_text(url: str) -> str:
    """
    Fetch the plain text content of a webpage by removing all HTML tags.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: Plain text content of the webpage.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')
        plain_text = soup.get_text(separator="\n")  # Use newline as separator for better readability
        return plain_text.strip()
    except requests.exceptions.RequestException as e:
        return f"Error fetching the webpage: {e}"

async def return_profiles_list(user_names_list: list, limit: int = 10):
    if not isinstance(user_names_list, list):
        user_names_list = [user_names_list]  # Ensure input is a list

    apify_client = ApifyClientAsync(token=os.getenv("APIFY_CLIENT"))
    apify_tasks_client = apify_client.tasks()

    apify_tasks: list[dict] = []
    for name in user_names_list:
        sanitized_name = sanitize_name(f"profiles-{name}")
        truncated_name = str(sanitized_name)[:50]  # Limit to 50 characters to include the prefix
        apify_task = await apify_tasks_client.create(
            name=f"profiles-{truncated_name}",
            actor_id="od6RadQV98FOARtrp",
            task_input={"keywords": [name], "limit": limit},  # Ensure limit is passed
            memory_mbytes=1024,
        )
        apify_tasks.append(apify_task)

    task_clients: list[TaskClientAsync] = [
        apify_client.task(task["id"]) for task in apify_tasks
    ]
    run_results = await asyncio.gather(
        *(client.call() for client in task_clients)  # Run all tasks
    )

    profiles = []
    for result in run_results:
        dataset_id = result.get("defaultDatasetId")
        async for item in apify_client.dataset(dataset_id).iterate_items():
            profiles.append(item)
            if len(profiles) >= limit:  # Enforce limit locally
                break

    return profiles

    # END FUNCTION HERE



async def return_profiles_contacts(user_names_list: dict, profile_links: list, limit: int = 3):
    client = ApifyClient("apify_api_gogvGT0ww4sI9rMKyBCbWBbgZyAbef44yQzp")

    run_input = {
        "profileUrls": profile_links
    }

    run = client.actor("2SyF0bVxmgGr8IVCZ").call(run_input=run_input)

    profile_details = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        profile_details.append(item)
    return profile_details

# END FUNCTION HERE
async def return_links(user_query: list, limit: int = 10):
    truncated_user_query = user_query[:1000]
    apify_client = ApifyClientAsync(token=os.getenv("APIFY_CLIENT"))

    run_input = {
        "search": truncated_user_query,
        "countryCode": "ng",  # Example: Set country code for Nigeria
    }

    run = apify_client.actor("Eva4hfW3EyWE7pVu2").call(run_input=run_input)
    dataset_client = apify_client.dataset(run["defaultDatasetId"])

    links = []
    offset = 0
    items_per_page = 5  # Number of items per batch

    while len(links) < limit:
        response = dataset_client.list_items(limit=items_per_page, offset=offset)
        items = response.items

        for item in items:
            if "url" in item:
                links.append(item["url"])
                if len(links) >= limit:
                    break

        if not items:
            break

        offset += items_per_page

    llm = LLM(model="gpt-4o-mini", temperature=0)
    import asyncio

    name_of_companies = await asyncio.to_thread(return_company_names, links)



    def extract_raw_list(website_links):
        for link in website_links:
            if isinstance(link, tuple) and len(link) > 1 and link[0] == 'raw':
                raw_content = link[1]
                try:
                    extracted_list = ast.literal_eval(raw_content)
                    if isinstance(extracted_list, list):
                        return extracted_list
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing 'raw' content: {e}")
        return [] 
        # END IN LINE FUNCTION HERE

    result = []
    ks = []

    real_names = extract_raw_list(name_of_companies)
    for link in real_names:
        if isinstance(link, tuple) and len(link) > 1 and isinstance(link[1], str):
            ks.append(StringKnowledgeSource(content=link[1]))
        elif isinstance(link, str):
            ks.append(StringKnowledgeSource(content=link))
        elif isinstance(link, list) and len(link) > 0:
            for item in link:
                if isinstance(item, str):
                    ks.append(StringKnowledgeSource(content=item))
                    break
        else:
            print(f"Skipping invalid link: {link}")

        if len(real_names) > 0:
            agent = Agent(
                role=f"Analyze these The information in your knowledge base. make a research and find out the names of the key players and decision makers for each of those companies.",
                goal="Provide a python list comprising of the names of the key decision makes in the companies. Also the links to their linkedin profiles.",
                backstory="You know how to make research and extract contact information and people data from data.",
                verbose=True,
                allow_delegation=True,
                tools=[custom_search_tool],
                llm=llm,
            )

            task = Task(
                description="You have been give a list of companies. For each of those companies, make a research and get me the names and linkedin profile links of the executives and key decision makers.",
                expected_output="Return a JSON object with the names and titles of key decision-makers. Be strict with your response. I do not need unneccesary information. I just need people's names and a list of their linked in profiles in a JSON",
                agent=agent,
                inputs={"data": name_of_companies},
            )

            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential,
                knowledge_sources=ks
            )
            result = crew.kickoff(inputs={"question": user_query})
            return result
        else:
            result = ["No information to display"]
            return result


# END FUNCTION HERE





def return_company_names(website_links: list) -> list:
    all_websites = []
    for link in website_links:
        a_link = fetch_plain_text(link)
        all_websites.append(a_link)

    ks = [StringKnowledgeSource(content=link) for link in all_websites]
    print(ks)
    agent = Agent(
        role=f"Analyze these The information in your knowledge base. Those are websites data. fish out the names of the organizations.",
        goal="Provide a python list comprising of the names of companies. I want you to be very strict about this. Try not to return any other word or reports. I just need the list of names",
        backstory="You know how to extract contact information and names from data.",
        verbose=True,
        allow_delegation=True,
        tools=[custom_search_tool],
        llm=llm,
    )

    task = Task(
        description="Analyze these The information in your knowledge base. Those are websites data. a list of the names of companies",
        expected_output="Return a JSON object with the names and titles of key decision-makers.",
        agent=agent,
        inputs={"data": all_websites},
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
        knowledge_sources=ks
    )

    result = crew.kickoff(inputs={"question": "return me the names of the companies in the list of data I gave to you"})
    return result



# END FUNCTION HERE



def extract_information(data):
    # Define regex patterns for phone number, email, and address
    phone_pattern = re.compile(r'\+?\d[\d -]{8,12}\d')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    address_pattern = re.compile(r'\d{1,5}\s\w+\s\w+')
    name_pattern = re.compile(r'[A-Z][a-z]+\s[A-Z][a-z]+')

    phone_number = phone_pattern.findall(data)
    email_address = email_pattern.findall(data)
    physical_address = address_pattern.findall(data)
    name = name_pattern.findall(data)

    return {
        "name": name,
        "phone_number": phone_number,
        "email_address": email_address,
        "physical_address": physical_address
    }




# First way to run this code:
# First we collect the prompt from the user from the front end api,
"""
    After which. We run the prompt to get the key words.
"""
# from langchain.chat_models import ChatAnthropic as LangchainChatAnthropic

class process_prompt():
    def __init__(self, prompt):
        self.prompt = prompt
        self.llm = LLM(model="gpt-4o-mini", temperature=0)
        
    async def extract_keywords(self):
        pr = self.prompt
        text_process_agent = Agent(
            role = f"Process this text -- {pr} and try to extract the keywords from it.  I am going to run a search with it on linked it. Return a maximum of four words. Do not separate the words with commas.",
            goal = "To find the most important words in sentence / prompt that can be used to run a search on social media networks",
            backstory = """
                    You are a precise and consice AI. Who filters large data to return the most important words. You love organizing keywords in a way that it makes the most possible sense
            """,
            verbose = True,
            allow_delegation= False,
            tools=[],
            llm=self.llm
            )
        
        
        process_text_task = Task(
                    description="Extracting keywords from the prompt to make a linkedin search. I need a maximum of four words.",
                    expected_output="A string of the keywords. Maintain a strict output of just the keywords and nothing else",
                    agent = text_process_agent,
                    inputs = {"data" : pr}
        )

        process_text_crew = Crew(
            agents = [text_process_agent],
            tasks=[process_text_task],
            verbose=True,
            process=Process.sequential,
        )

        result = process_text_crew.kickoff(inputs = {"question" : pr})
        # print("my response is: ", result)
        return result
    

    def run(self, mode):
            if mode == "process":
                return self.extract_keywords()
            


class Contact_Details():
    def __init__(self, keywords, prompt):
        self.keywords = keywords
        self.prompt = prompt
        self.apify_api_key = ApifyClientAsync(token=os.getenv("APIFY_CLIENT"))
        self.client = ApifyClient(self.apify_api_key)
        self.appKey = "UodqxkdCqDp5KZqZE"

    async def fetch_contacts_from_linkedin(self, countryCode):
        try:
            run_input = {
                "keyword": self.keywords,
                "platform": "LinkedIn",
                "country": countryCode,
                "maxResults": 40,
                "proxyConfiguration": {"useApifyProxy": True},
            }
            run = self.client.actor(self.appKey).call(run_input=run_input)
            all_items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())
            return self.clean_up_dataset(all_items)
        except Exception as e:
            print(f"Error fetching contacts: {e}")
            raise

    def clean_up_dataset(self, data):
        formatted_data = []
        for item in data:
            formatted_data.append({
                "keywords": item.get('keywords'),
                "countryCode": item.get('countryCode'),
                "phoneNumber": item.get('phoneNumber'),
                "title": item.get('title'),
                "url": item.get('url'),
                "text": item.get('text'),
                "platform": item.get('platform'),
            })
        return formatted_data
    
class ContactDetail(BaseModel):
    name: str
    phone_number: str
    email_address: str
    physical_address: str
    company_name: str
    company_favicon: str
    individual_position: str
    ai_insight: str


class ContactDetails(BaseModel):
    contacts: list[ContactDetail]

class get_details_using_ai:
    def __init__(self, data):
        self.data = str(data)
        self.llm = LLM(model="gpt-4o-mini", temperature=0)
        chars_to_remove = ['/', '\\', '\n']
    
        # Create a translation table
        translation_table = str.maketrans('', '', ''.join(chars_to_remove))
        
        # Translate the input string using the translation table
        self.cleaned_string = self.data.translate(translation_table)
        self.cleaned_string = str(self.cleaned_string)

    async def get_user_information(self, userInfo):
        print(f"raw data is: {userInfo}")
        processed_data = self.cleaned_string
        print("Processed data: ", processed_data)
        values_list = []
        print("items in userInfo ", userInfo)
        

        for item in userInfo:
            # Convert the string representation of the dictionary to an actual dictionary
            dictionary = ast.literal_eval(item)
            # Extract the values and extend the values_list
            values_list.extend(dictionary.values())

        print("Items in values list : ", values_list)
        def extract_raw_list(website_links):
            extracted_items = []
            for link in website_links:
                if isinstance(link, tuple) and len(link) > 1 and link[0] == 'raw':
                    raw_content = link[1]
                    try:
                        extracted_list = ast.literal_eval(raw_content)
                        if isinstance(extracted_list, list):
                            extracted_items.extend(extracted_list)
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing 'raw' content: {e}")
            return extracted_items
            # END IN LINE FUNCTION HERE

        result = []
        ks = []

        real_names = extract_raw_list(userInfo)
        for link in real_names:
            if isinstance(link, tuple) and len(link) > 1 and isinstance(link[1], str):
                ks.append(StringKnowledgeSource(content=link[1]))
            elif isinstance(link, str):
                ks.append(StringKnowledgeSource(content=link))
            elif isinstance(link, list) and len(link) > 0:
                for item in link:
                    if isinstance(item, str):
                        ks.append(StringKnowledgeSource(content=item))
                        break
                    else:
                        print(f"Skipping invalid link: {link}")

        try:
            values_list = str(values_list)
            info_agent = Agent(
                role=f"Take data list - {values_list}, study it, and extract the following information:"
                    "- The name of the person"
                    "- name of the business / company"
                    "- Position of the individual"
                    "- Their Phone number,"
                    "- Their email address,"
                    "- Company favicon link"
                    "- Possition of individual"
                    "- Your insight about what this company does"
                    "- Their physical address. Be very precise with your feedback. Do not return names like John Doe or anything uncertain. In the event where you do not know a name, simply return an empty string. Using your tools, also do your best to come up with the: company's favicon url i.e google google the company name or the name of the person, get the link and extract the favicon. For any of the datasets where information is not complete, fo the following: If the individual's name is there and the company name isn't, Google the person's name and ask what company he works for. If the company's name and the role are there, but the person's name is not there, make a google search and ask what the person's name is eg: who is the ceo of standard oil. Also if you note that the name of the individual is a sole proprietor or just an individual, not an organization. In the individual_position field, just put entreprenuera and the company name should be private entreprise.",
                goal="Produce a clear and formatted output of a person's information from the provided data set.",
                backstory=(
                    "You have the ability to cut through clutter and provide the most valuable information."
                ),
                verbose=True,
                allow_delegation=True,
                tools=[google_search_tool, apify_get_profile_details_tool],
                llm=self.llm
            )



            process_info_agent = Agent(
                    role = f"Take the information returned by the previous agent and try to fill in the blank spaces. For every set of information returned, I am expecting the following response: - A link to the company website's favicon. So if the name of the organization or company is given, use your tool and do a google search to get the official website of the company. and return that link's favicon. If the name of the company is not available, find out if the name of the key decision maker is available and make a google search to find out where he works and return the name of the company and the link to the favicon too.",
                    goal = "To do a research in order to get detailed information about the details provided to you. Look for names of the companies, names of the executives, links to the company logo / favicon. If you can't find their favicon, return the link to their linkedin profile picture",
                    backstory = (
                        "You are a highly detailed professional research agent who knows how to get out important needed information from the internet."
                    ),
                    verbose = True,
                    allow_delegation=True,
                    tools=[google_search_tool, apify_get_profile_details_tool],
                    llm=self.llm
            )

            # Validating_agent = Agent(
            #     role = "To Take data returned by your other agent and to do a search and to return acurate information."
            # )
            info_task = Task(
                description="Extract the phone_number, email_address, physical address, and name from the provided data. Do not make up any information. If you cannot find something, try to find it or else leave it empty",
                expected_output="Return a JSON object or Python dictionary with the extracted information.",
                agent=info_agent,
                inputs={"data": processed_data},
                # output_json=ContactDetails  # Set the output format to JSON using a Pydantic model
            )

            process_response_task = Task(
                description="Make a research with the information you have received to achieve the goals already outlined for you.",
                expected_ouput = "Return a JSON object with the complete information given to you by making researches with the tools in your tools box and others too",
                agent = process_info_agent,
                inputs = {"data" : processed_data},
                output_json = ContactDetails,
                context=[info_task]
            )

            info_crew = Crew(
                agents=[info_agent, process_info_agent],
                tasks=[info_task, process_response_task],
                verbose=True,
                process=Process.sequential,

            )

            ai_result = info_crew.kickoff(inputs={"question": values_list})
            
            # Convert CrewOutput to a dictionary if possible
            if hasattr(ai_result, "to_dict"):
                return ai_result.to_dict()
            
            # Otherwise, manually extract relevant information
            return {"result": str(ai_result)}
        

        except Exception as e:
            print(f"Error in _get_user_information: {e}")
            raise

class validate_ai_response():

    def __init__(self, initial_response):
        return None



