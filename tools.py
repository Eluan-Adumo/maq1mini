import difflib
import re
from bs4 import BeautifulSoup
from crewai.tools import tool
# from crewai_tools import SerperDevTool
import requests
from apify_client import ApifyClient # type: ignore
from urllib.parse import urljoin
import ast
import os
from crewai import Agent, Crew, Task


@tool
def google_search_tool(query: str, api_key: str, cse_id: str, num_results: int = 10):
	"""
	Perform a Google Custom Search using Google Custom Search JSON API.

	Args:
		query (str): The search query.
		api_key (str): Your Google API Key.
		cse_id (str): Your Custom Search Engine ID.
		num_results (int): Number of results to retrieve.
		date_restrict (str): Restrict results to a specific time range (e.g., "d[7]" for the last 7 days, "m[1]" for the last month, "y[1]" for the last year).

	Returns:
		dict: A dictionary of search results.
	"""

	url = "https://www.googleapis.com/customsearch/v1"
	params = {
		"q": query,
		"key": api_key,
		"cx": cse_id,
		"num": num_results,
		# "dateRestrict": date_restrict
	}

	response = requests.get(url, params=params)
	response.raise_for_status()
	results = response.json()

	# Extract all the relevant data
	search_results = [
		{
			"title": item.get("title"),
			"link": item.get("link"),
			"snippet": item.get("snippet"),
		}
		for item in results.get("items", [])
	]

	return {"results": search_results}



@tool
def crawl_website(url: str):
    """
    Crawls a website to extract key data, including the page title, all links, 
    headings (H1, H2, H3), and potential logo images.

    Args:
        url (str): The URL of the website to crawl.

    Returns:
        dict: A dictionary containing the following crawled data:
            - "title": The title of the webpage.
            - "links": A list of all hyperlinks (anchor tags) on the webpage.
            - "headings": A dictionary containing all headings (H1, H2, H3).
            - "images": A list of image URLs that likely correspond to logos (based on patterns in the URL).
    """
    try:
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    except Exception as e:
        return {"error": str(e)}

    # Parse the webpage content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the title of the webpage
    title = soup.title.string if soup.title else "No title found"

    # Extract all hyperlinks (anchor tags)
    links = [a['href'] for a in soup.find_all('a', href=True)]

    # Extract all headings (H1, H2, H3)
    headings = {
        "h1": [h.get_text().strip() for h in soup.find_all('h1')],
        "h2": [h.get_text().strip() for h in soup.find_all('h2')],
        "h3": [h.get_text().strip() for h in soup.find_all('h3')]
    }

    # Extract all image tags (img) and look for logos
    img_tags = soup.find_all('img', src=True)
    img_urls = []

    # Filter out potential logo images
    for img in img_tags:
        img_url = img['src']
        # Make sure to handle relative URLs using urljoin
        img_url = urljoin(url, img_url)

        # Some common patterns for logo images (you can extend this list)
        if re.search(r'(logo|brand|header|favicon)', img_url, re.IGNORECASE):
            img_urls.append(img_url)

    # Return the crawled data as a dictionary
    return {
        "title": title,
        "links": links,
        "headings": headings,
        "images": img_urls
    }




@tool
def google_search_tool_decorated(query: str):
	"""
	Perform a Google Custom Search using Google Custom Search JSON API.

	Args:
		query (str): The search query.
		api_key (str): Your Google API Key.
		cse_id (str): Your Custom Search Engine ID.
		num_results (int): Number of results to retrieve.

	Returns:
		dict: A dictionary of search results.
	"""
	# return google_search_tool(query, api_key="AIzaSyDSXStmZvKfp6ww1_cmWVltbyYPBZdK_yE", cse_id="a673099502c384a2f", date_restrict="d[7]")

	return google_search_tool(query, api_key="AIzaSyDSXStmZvKfp6ww1_cmWVltbyYPBZdK_yE", cse_id="a673099502c384a2f")




@tool
def apify_get_profiles_tool(keywords: list, limit: int = 3):
	"""
	Fetch profiles using Apify Actor.

	Args:
		keywords (list): List of keywords to search for profiles.
		limit (int): Number of profiles to retrieve.

	Returns:
		list: A list of profiles.
	"""
	client = ApifyClient("apify_api_gogvGT0ww4sI9rMKyBCbWBbgZyAbef44yQzp")

	run_input = {
		"action": "get-profiles",
		"keywords": keywords,
		"isUrl": True,
		"isName": True,
		"limit": limit,
	}

	run = client.actor("od6RadQV98FOARtrp").call(run_input=run_input)

	profiles = []
	for item in client.dataset(run["defaultDatasetId"]).iterate_items():
		profiles.append(item)

	return profiles
     
	# async def return_numbers_emails_dps(profile):
    #       get_details_agent = Agent(
			
	# 	  )


@tool
def extract_user_profile(profile_data: list):
    """
    Extracts user profile information from raw profile data.

    Args:
        profile_data (list): The list of all the users' profile data.

    Returns:
        list: A list of dictionaries with all the dictionaries containing the following keys:
            - "user_name": The user name extracted from the profile data (or None if not found).
            - "email address": The email address extracted from the profile data (or None if not found).
            - "phone number": The phone number extracted from the profile data (or None if not found).
            - "profile photo": The URL of the profile photo extracted from the profile data (or None if not found).

    Notes:
        - Email addresses, phone numbers, and profile photo URLs are extracted using regular expressions.
        - If any piece of information is not found, it defaults to None.
    """
    final_result = []

    # Helper function to extract email
    def extract_email(data):
        email_regex = r"[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}"  # Basic email regex
        match = re.search(email_regex, data)
        return match.group(0) if match else None

    # Helper function to extract phone number
    def extract_phone(data):
        phone_regex = r"\b\d{10,15}\b"  # Matches phone numbers with 10-15 digits
        match = re.search(phone_regex, data)
        return match.group(0) if match else None

    # Helper function to extract profile photo URL
    def extract_photo_url(data):
        url_regex = r"https?://[\w./%-]+(?:\.(?:jpg|jpeg|png|gif|bmp))"  # Basic image URL regex
        match = re.search(url_regex, data)
        return match.group(0) if match else None

    # Helper function to extract user name
    def extract_user_name(data):
        name_regex = r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b"  # Matches names with capitalized words
        match = re.search(name_regex, data)
        return match.group(0) if match else None

    # Process each profile in profile_data
    for profile in profile_data:
        result = {
            "user_name": extract_user_name(profile),
            "email address": extract_email(profile),
            "phone number": extract_phone(profile),
            "profile photo": extract_photo_url(profile)
        }
		
        final_result.append(result)

    print("The extracted data is: ", final_result)
    return final_result


@tool
def apify_get_profile_details_tool(profile_urls: list):
    """
    Fetch LinkedIn profile details using Apify Actor.

    Args:
        profile_urls (list): List of LinkedIn profile URLs (e.g., ["https://linkedin.com/in/..."]).

    Returns:
        list: A list of dictionaries containing profile details.
              If an error occurs, a dictionary with an "error" key is returned.
    """


    # Validate input
    if not isinstance(profile_urls, list) or not all(isinstance(url, str) for url in profile_urls):
        raise ValueError("profile_urls must be a list of valid strings (URLs).")

    try:
        # Initialize Apify Client with API key from environment
        client = ApifyClient("apify_api_gogvGT0ww4sI9rMKyBCbWBbgZyAbef44yQzp")

        # Define the input for the actor
        run_input = {"profileUrls": profile_urls}

        # Trigger the actor and fetch the dataset
        run = client.actor("2SyF0bVxmgGr8IVCZ").call(run_input=run_input)

        if not run.get("defaultDatasetId"):
            return {"error": "No dataset ID found in the run response."}

        profile_details = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            profile_details.append(item)
            print("Profile data: ", item)
		
        return profile_details

    except Exception as e:
        return {"error": f"Failed to fetch profile details: {str(e)}"}

@tool 
def custom_search_tool(query: list, max_results: int = 10):
	"""
	Generate query from the command and run a search with this tool

	Args:
		query (list): query to search with.
		max_results (int): Maximum number of search results to retrieve.

	Returns:
		list: A list of website links and other details.
	"""
	client = ApifyClient("apify_api_gogvGT0ww4sI9rMKyBCbWBbgZyAbef44yQzp")
	run_input = {
		"search": query,
		"countryCode": "",
		"maxResults": min(max_results, 10)  # Ensure max_results does not exceed 10
	}

	# Run the Actor and wait for it to finish
	run = client.actor("Eva4hfW3EyWE7pVu2").call(run_input=run_input)

	# Fetch and print Actor results from the run's dataset (if there are any)
	results = []
	for item in client.dataset(run["defaultDatasetId"]).iterate_items():
		results.append(item)

	return results

