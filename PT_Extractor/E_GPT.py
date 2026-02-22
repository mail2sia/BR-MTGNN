import os
import openai
import wandb
import random
from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch
import json
import sys
import csv
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

# configs

openai.api_key = "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"  # place your key here.

## Load configuration
with open("config.json") as con_file:
    config = json.load(con_file)
api_key = config["apikey"]

def getDefaultFirefoxOptions():
    options = Options()
    options.add_argument('--headless')
    options.add_argument("-profile")
    options.add_argument("--marionette-port")
    options.add_argument("2828")
    options.add_argument("--headless")
    options.add_argument("devtools.selfxss.count", 100)


options = Options()
options.add_argument('--headless')
driver = webdriver.Firefox(options=options)


# computes the number of words in a keyword/phrase
def len_kw(keyword):
    keyword_list = keyword.split(' ')
    return len(keyword_list)


# checks whether a keyword (or phrase) is the first part of a list of strings
def keyword_in_list(keyword, text):
    keyword_list = keyword.split(' ')

    if len(keyword_list) > len(text):
        return False
    for i in range(len(keyword_list)):
        if keyword_list[i] != text[i]:
            return False

    return True


# computes the smallest distance between a keyword and any term in a list of solution synonyms.
# if no such distance found, returns the length of the given text
def compute_min_distance(key_word, abstract):
    with open('solution_synonyms.txt') as file:
        terms = [line.rstrip() for line in file]

    words = abstract.upper().replace('-', ' ').replace('\n', ' ').replace('.', '').replace(',', '').replace(';',
                                                                                                            '').replace(
        '?', '').split(' ')
    keyword = key_word.upper().replace('-', ' ').replace('\n', ' ').replace('.', '').replace(',', '').replace(';',
                                                                                                              '').replace(
        '?', '')
    # print('words list=',words)

    min_distance = 999999999
    distance = 0
    for i in range(len(words)):
        if keyword_in_list(keyword, words[i:]):  # keyword here?
            # iterate backwards until you find the term (e.g.,"solution")
            for j in range(i - 1, -1, -1):
                if words[j] in terms:
                    distance = i - j
                    if distance < min_distance:
                        min_distance = distance
                    break

            # iterate forward until you find the term (e.g.,"overcome")
            for j in range(i + len_kw(keyword), len(words)):
                if words[j] in terms:
                    distance = j - ((i + len_kw(keyword)) - 1)
                    if distance < min_distance:
                        min_distance = distance
                    break
    if min_distance == 999999999:
        print('DISTANCE NOT FOUND!')
        return len(abstract)

    return min_distance


def is_exclude_solution(solution, disease):
    with open('excluded_keywords.txt') as file:
        lines = [line.rstrip() for line in file]

    solution = solution.upper().replace('-', ' ')
    exclude = False
    for line in lines:
        if line.upper().replace('-', ' ') == solution:
            exclude = True
            break
        if line.upper().replace('-', ' ') + 'S' == solution:
            exclude = True
            break
        if line + ' disease'.upper().replace('-', ' ') == solution:
            exclude = True
            break
        if line + ' disease'.upper().replace('-', ' ') + 'S' == solution:
            exclude = True
            break

    if solution == disease.upper().replace('-', ' ') + ' DETECTION' or solution == disease.upper().replace('-',
                                                                                                           ' ') + ' DISEASE DETECTION':
        exclude = True
    if solution == disease.upper().replace('-', ' ') + ' PREVENTION' or solution == disease.upper().replace('-',
                                                                                                            ' ') + ' DISEASE PREVENTION':
        exclude = True
    if solution == disease.upper().replace('-', ' ') + ' PROTECTION' or solution == disease.upper().replace('-',
                                                                                                            ' ') + ' DISEASE PROTECTION':
        exclude = True
    if solution == disease.upper().replace('-', ' ') + ' MITIGATION' or solution == disease.upper().replace('-',
                                                                                                            ' ') + ' DISEASE MITIGATION':
        exclude = True
    if solution == disease.upper().replace('-', ' ') + ' DEFENSE' or solution == disease.upper().replace('-',
                                                                                                         ' ') + ' DISEASE DEFENSE':
        exclude = True
    if solution == disease.upper().replace('-', ' ') + ' DEFENCE' or solution == disease.upper().replace('-',
                                                                                                         ' ') + ' DISEASE DEFENCE':
        exclude = True
    if solution.endswith('DISEASE') or solution.endswith('DISEASES') or solution.endswith(
            'SECURITY') or solution.endswith(disease) or solution.endswith(disease + 'S') or solution.endswith(
            'THREAT') or solution.endswith('THREATS') or solution.endswith('PROTECTION') or solution.endswith(
            'DETECTION') or solution.endswith('PREVENTION') or solution.endswith('MITIGATION'):
        exclude = True

    return exclude


# This method collects n research abstracts (with title and keywords) from Elsevier related to a given disease.
# The abstract should include mitigation related keywords/pertinent technologies.
# It is recommended that you clear your Firefox browser before running this code.
def query_Elsevier_for_disease_solutions_abstracts(disease, abs_N=10):
    """
    Fetch research abstracts from Scopus API related to a given rare mental disease.
    """
    url = f"https://api.elsevier.com/content/search/scopus?query={disease}&apiKey={api_key}"
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        abstracts = []

        # Extract abstracts
        for entry in data.get("search-results", {}).get("entry", [])[:abs_N]:
            abstract = entry.get("dc:description", "")
            if abstract:
                abstracts.append(abstract)

        return abstracts
    else:
        print("Error:", response.status_code, response.text)
        return []

# Given an abstract and an disease type, this method prompts GPT3 to extract technology solutions pertinent to the given disease from the abstract.
def prompt_GPT3_to_extract_pertinent_technologies(disease, abstract):
    """
    Prompts GPT-3 to extract medical, technological, and therapeutic solutions for a rare mental disease.
    """
    example = """Q: Extract the list of medical treatments and technologies for Schizophrenia from the following text

    [Example Abstract]

    Solutions: Antipsychotic drugs, Cognitive Behavioral Therapy, Electroconvulsive Therapy, Brain Stimulation, Digital Health Tools

    Q: Extract the list of medical treatments and technologies for PTSD from the following text

    [Example Abstract]

    Solutions: Trauma-focused CBT, Virtual Reality Exposure Therapy, SSRIs, Eye Movement Desensitization and Reprocessing, Telemedicine
    """

    gpt_prompt = f"{example}\nQ: Extract the list of medical treatments and technologies for {disease} from the following text\n\n{abstract}\n\nSolutions:"

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=gpt_prompt,
        temperature=1,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    answer = response["choices"][0]["text"]

    # Format answer as a comma-separated list
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f'Extract the list from the following text, and separate items by comma and space:\n\n{answer}',
        temperature=0,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    answer_list = response["choices"][0]["text"].replace("\n", ", ").split(", ")
    answer_list = list(dict.fromkeys([s.strip() for s in answer_list if s.strip()]))

    return answer_list


# returns top N relevant technology solutions to a given disease
def iterative_prompt(disease_list, N, abs_N):
    """
    Iterates over the list of rare mental diseases to extract N most relevant medical technologies.
    """
    for disease in disease_list:
        abstracts = query_Elsevier_for_disease_solutions_abstracts(disease, abs_N)
        tech_rank = {}

        for i, abstract in enumerate(abstracts):
            tech_keywords = prompt_GPT3_to_extract_pertinent_technologies(disease, abstract)
            print(f"Keywords for {disease} (abstract {i+1}): {tech_keywords}")
            time.sleep(40)  # To avoid rate limit errors

            for tech in tech_keywords:
                if tech not in tech_rank:
                    tech_rank[tech] = 1
                else:
                    tech_rank[tech] += 1

        # Sort and get top N technologies
        sorted_tech = sorted(tech_rank.items(), key=lambda x: x[1], reverse=True)
        top_tech = [t[0] for t in sorted_tech[:N]]

        print(f"\nTop Technologies for {disease}:\n", top_tech)

        with open("E_GPT.txt", "a") as file:
            file.write(disease.upper() + ", " + ", ".join(top_tech) + "\n")


import time
import openai
import requests

# List of rare mental diseases
diseases = [
    "Reactive Attachment Disorder", "Pyromania", "Othello Syndrome", "Social Engagement Disorder",
    "Schizoaffective Disorder", "Rett Syndrome", "Catatonia", "Olfactory Reference Disorder",
    "Hypergraphia", "Wendigo Psychosis", "Body Integrity Dysphoria", "Agoraphobia", "Ablutophobia",
    "Delusional Disorder", "Hoarding Disorder", "Impulse Control Disorder", "Postpartum Psychosis",
    "Separation Anxiety Disorder", "Brief Psychotic Disorder", "Schizotypal Personality Disorder",
    "Selective Mutism", "Kleine-Levin Syndrome", "Schizotypal Disorder", "Capgras Syndrome",
    "Apotemnophilia", "Depersonalization-Derealization Disorder", "Circadian Rhythm Sleep-Wake Disorder",
    "Acute and Transient Psychotic Disorder", "Rumination Disorder", "Cyclothymic Disorder",
    "Walking Corpse Syndrome", "Diogenes Syndrome", "Landau-Kleffner Syndrome", "Psychogenic Amnesia",
    "Cotard Delusion", "PURA Syndrome", "Stendhal Syndrome", "Fregoli Delusion", "Reduplicative Amnesia",
    "Histrionic Personality Disorder", "Autocannibalism", "Ganser Syndrome", "Kleptomania",
    "Jerusalem Syndrome", "Boanthropy", "Secondary Personality Change", "Hallucinogen-Induced Psychotic Disorder",
    "Clinical Lycanthropy", "Bachmann-Bupp Syndrome", "Factitious Disorder", "Hyperthymesia",
    "Paramnesia", "Palilalia"
]

# Number of pertinent technologies to extract
N = 10
# Number of abstracts to fetch per disease
abs_N = 100


def query_Elsevier_for_disease_solutions_abstracts(disease, abs_N=100):
    """
    Fetch research abstracts (or full-text links) from Scopus API for a given disease.
    """
    url = f"https://api.elsevier.com/content/search/scopus?query=TITLE({disease}) OR ABS({disease}) OR KEY({disease}) AND PUBYEAR > 2000&apiKey={api_key}"
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        print(f"Scopus API Response for {disease}: {data}")  # Debugging output

        abstracts = []
        for entry in data.get("search-results", {}).get("entry", [])[:abs_N]:
            abstract = entry.get("dc:description", "")

            if not abstract:
                doi = entry.get("prism:doi", None)
                if doi:
                    print(f"🔍 Fetching full-text for {disease} (DOI: {doi})")
                    full_text_abstract = fetch_full_text_abstract(doi)
                    if full_text_abstract:
                        abstract = full_text_abstract

            if abstract:
                abstracts.append(abstract)

        if not abstracts:
            print(f"⚠️ No abstracts found for {disease}. Expanding query may be needed.")
        return abstracts
    else:
        print(f"❌ Error fetching abstracts for {disease}: {response.status_code} {response.text}")
        return []


def fetch_full_text_abstract(doi):
    """
    Fetch full-text abstract using Elsevier's Article Retrieval API.
    """
    url = f"https://api.elsevier.com/content/article/doi/{doi}?apiKey={api_key}"
    headers = {"Accept": "application/json", "X-ELS-APIKey": api_key}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get("full-text-retrieval-response", {}).get("coredata", {}).get("dc:description", "")
    return None


def fetch_full_text_abstract(doi):
    """
    Fetch full-text abstract using DOI link.
    """
    if not doi:
        return None  # Return None if DOI is missing

    # Convert DOI to a valid URL
    doi_url = f"https://doi.org/{doi}"
    headers = {"Accept": "application/json"}

    try:
        response = requests.get(doi_url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return data.get("abstract", None)  # Extract abstract if available
        else:
            print(f"⚠️ Unable to fetch full-text abstract for DOI: {doi} (Status Code: {response.status_code})")
            return None

    except Exception as e:
        print(f"❌ Error fetching full-text abstract: {e}")
        return None


def prompt_GPT3_to_extract_pertinent_technologies(disease, abstract):
    """
    Prompts GPT-3 to extract medical, technological, and therapeutic solutions for a rare mental disease.
    """
    gpt_prompt = f"""
    Q: List the relevant medical treatments and technologies for {disease}.

    Example 1:
    Schizophrenia -> Antipsychotic drugs, Cognitive Behavioral Therapy, Brain Stimulation, Digital Health Tools

    Example 2:
    PTSD -> Trauma-focused CBT, Virtual Reality Exposure Therapy, SSRIs, Eye Movement Desensitization

    Now, list the technologies for {disease} from the following text:

    {abstract}

    Solutions:
    """

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=gpt_prompt,
        temperature=0.7,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    answer = response["choices"][0]["text"].strip()

    # Reformat the output as a clean, comma-separated list
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Extract the list from the following text, separating items with commas:\n\n{answer}",
        temperature=0,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    answer_list = response["choices"][0]["text"].replace("\n", ", ").split(", ")
    answer_list = list(dict.fromkeys([s.strip() for s in answer_list if s.strip()]))

    return answer_list


def iterative_prompt(disease_list, N, abs_N):
    """
    Iterates over the list of rare mental diseases to extract N most relevant medical technologies.
    """
    for disease in disease_list:
        print(f"\n🔍 Fetching abstracts for {disease}...")
        abstracts = query_Elsevier_for_disease_solutions_abstracts(disease, abs_N)

        tech_rank = {}
        for i, abstract in enumerate(abstracts):
            tech_keywords = prompt_GPT3_to_extract_pertinent_technologies(disease, abstract)
            print(f"Keywords for {disease} (abstract {i+1}): {tech_keywords}")
            time.sleep(40)  # To avoid API rate limits

            for tech in tech_keywords:
                if tech not in tech_rank:
                    tech_rank[tech] = 1
                else:
                    tech_rank[tech] += 1

        # Sort and get top N technologies
        sorted_tech = sorted(tech_rank.items(), key=lambda x: x[1], reverse=True)
        top_tech = [t[0] for t in sorted_tech[:N]]

        print(f"\n🧠 Top Technologies for {disease}: {top_tech}")

        # ✅ Write to file with UTF-8 encoding to prevent UnicodeEncodeError
        with open("E_GPT.txt", "a", encoding="utf-8") as file:
            file.write(disease.upper() + ", " + ", ".join(top_tech) + "\n")

if __name__ == "__main__":
    N = 10  # Number of pertinent technologies to extract
    abs_N = 50  # Number of abstracts to fetch per disease

    # Run the function for testing
    iterative_prompt(diseases, N, abs_N)
