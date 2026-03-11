# **PTs Extraction for Rare Mental Diseases (RMD)**
This repository contains Python implementations of two algorithms, **Extractive-GPT (E-GPT)** and **Direct-GPT (D-GPT)**, designed for extracting **Pertinent Technologies (PTs)** relevant to **Rare Mental Diseases (RMDs)**. Both algorithms interact with the **GPT-3 model** to identify technologies associated with each disease, leveraging different approaches for extraction and ranking.

## **Overview**
The algorithms work as follows:
- **E-GPT**: Extracts PTs by first retrieving **research abstracts from Elsevier’s database**, then prompting GPT to extract the relevant PTs from these abstracts. The extracted PTs are ranked based on frequency and proximity to key terms in the abstracts.
- **D-GPT**: Directly asks GPT a predefined question (e.g., *"What are the pertinent technologies relevant to [RMD]?"*) without leveraging external research abstracts.

Both algorithms filter out unwanted terms using a predefined **exclusion list**, stored in the file *excluded_keywords.txt*. Additionally, to enhance the quality of extracted PTs, the GPT model is provided with an example question and answer.

## **E-GPT**
- Implementation: **E-GPT.py**  
- Output file: **E-GPT.txt**  
- Uses: Elsevier API + GPT-3 for PT extraction from research articles.

## **D-GPT**
- Implementation: **D-GPT.py**  
- Output file: **D-GPT.txt**  
- Uses: Direct GPT-3 interaction for retrieving PTs.

## **Final Validation**
- Ask ChatGPT, Gemini and Claude to finalise the list

## **Dataset & Integration**
This extraction process contributes to a structured **forecasting model**, where the extracted PTs are used to **analyze trends and predict the technological landscape for RMDs** over the next three years. The extracted data integrates with **Graph Neural Networks (GNNs)** for time-series forecasting.

## **Notes/Requirements**
- **API Keys Required**:
  - OpenAI API key should be placed in the first line of code.
  - Elsevier API key should be stored in **config.json** to enable research article retrieval.
- **Prerequisites**:
  - Install required Python libraries using `pip install -r requirements.txt`
  - Ensure network access for API interactions with OpenAI and Elsevier.
