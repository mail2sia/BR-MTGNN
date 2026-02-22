import os
import time
import openai
import wandb
import random
import json
import requests
from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Load API keys from config.json
with open("config.json") as con_file:
    config = json.load(con_file)
    api_keys = config["apikey"]  # List of API keys


def get_random_api_key():
    """Select a random API key from the list."""
    return random.choice(api_keys)


openai.api_key = "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" #place your key here.


def prompt_gpt3_for_disease_solutions(disease):
    """Generates a list of research solutions (pertinent technologies) for treating a rare mental disease."""

    note = "Provide 10 answers. Please do not write any additional requests. Only answer the question I asked."

    # Example prompt to guide GPT
    example = (
            "Q. Give me the list of research solutions keywords for treating Alice in Wonderland Syndrome." + note +
            "\n\n A: Cognitive Behavioral Therapy (CBT), Neurofeedback Systems, AI-Powered Mental Health Platforms, Virtual Reality Therapy, Transcranial Magnetic Stimulation, Emotional Support Platforms, Digital Journaling Apps, AI-Based Stress Detection Platforms, Psychoeducation Tools, Deep Brain Stimulation."
    )

    # Construct GPT-3.5 request prompt
    gpt_prompt = example + f"Q. Give me the list of research solutions keywords for treating {disease}." + note + "\n\n A:"

    # Make API request to GPT-3.5
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=gpt_prompt,
        temperature=1,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    print("RESPONSE:")
    print(response['choices'][0]['text'])

    answer = response['choices'][0]['text']

    # Make another request to extract keywords properly
    response_cleaned = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Extract the list of solutions keywords from the following text and separate the items by comma and space:\n\n{answer}",
        temperature=0,
        max_tokens=2048,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    answer = response_cleaned['choices'][0]['text']

    # Process extracted PTs
    # 1. Remove numbering
    if len(answer) > 1 and str(answer[0]).isdigit() and answer[1] == ".":
        for i in range(30):
            answer = answer.replace(str(i + 1) + ". ", "")

    # 2. Replace dash in the middle of terms by space
    for i in range(len(answer)):
        if answer[i] == "-" and i > 0 and str(answer[i - 1]).isalpha():
            answer = answer[:i] + " " + answer[i + 1:]

    # 3. Replace dash with empty string elsewhere and format output
    answer_list = answer.replace("-", "").replace("\n", ", ").split(", ")
    return [s.strip() for s in answer_list if s.strip() != ""]


def is_exclude_solution(solution, attack):
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
        if line + ' attack'.upper().replace('-', ' ') == solution:
            exclude = True
            break
        if line + ' attack'.upper().replace('-', ' ') + 'S' == solution:
            exclude = True
            break

    if solution == disease.upper().replace('-', ' ') + ' TREATMENT' or solution == disease.upper().replace('-',
                                                                                                           ' ') + ' DISEASE TREATMENT':
        exclude = True
    if solution == disease.upper().replace('-', ' ') + ' PREVENTION' or solution == disease.upper().replace('-',
                                                                                                            ' ') + ' DISEASE PREVENTION':
        exclude = True
    if solution == disease.upper().replace('-', ' ') + ' MANAGEMENT' or solution == disease.upper().replace('-',
                                                                                                            ' ') + ' DISEASE MANAGEMENT':
        exclude = True
    if solution == disease.upper().replace('-', ' ') + ' DIAGNOSIS' or solution == disease.upper().replace('-',
                                                                                                           ' ') + ' DISEASE DIAGNOSIS':
        exclude = True
    if solution == disease.upper().replace('-', ' ') + ' AWARENESS' or solution == disease.upper().replace('-',
                                                                                                           ' ') + ' DISEASE AWARENESS':
        exclude = True
    if solution == disease.upper().replace('-', ' ') + ' DISORDER' or solution == disease.upper().replace('-',
                                                                                                          ' ') + ' DISEASE DISORDER':
        exclude = True
    if solution == disease.upper().replace('-', ' ') + ' RESEARCH' or solution == disease.upper().replace('-',
                                                                                                          ' ') + ' DISEASE RESEARCH':
        exclude = True

    # General exclusions for common medical terms
    if solution.endswith('DISEASE') or solution.endswith('DISEASES') or solution.endswith(
            'TREATMENT') or solution.endswith('THERAPY') or solution.endswith('DIAGNOSIS') or solution.endswith(
        'PREVENTION') or solution.endswith('AWARENESS') or solution.endswith('RESEARCH') or solution.endswith(
        'MANAGEMENT'):
        exclude = True

    return exclude


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

done = []

for disease in diseases:
    print('ASKING DIRECT Q FOR:', disease)
    all_solutions = prompt_gpt3_for_disease_solutions(disease)
    solutions = []
    for s in all_solutions:
        if len(solutions) < 10 and not is_exclude_solution(s, disease):
            solutions.append(s)

    print('PTs:\n', solutions, '\n')

    with open('D_GPT.txt', 'a') as the_file:
        the_file.write(disease.upper() + ', ')
        for i, v in enumerate(solutions):
            the_file.write(v.upper())
            if i == len(solutions) - 1:
                the_file.write('\n')
            else:
                the_file.write(', ')
    time.sleep(40)
with open('D_GPT.txt', 'a') as the_file:
    the_file.write('\n')