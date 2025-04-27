# from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
# import requests
# import json
# import ibm_boto3
# from ibm_botocore.client import Config
# from ibm_botocore.config import Config as BotocoreConfig
# import urllib.parse
# import os
# from datetime import datetime
# from dotenv import load_dotenv
# import logging
# import time
# from concurrent.futures import ThreadPoolExecutor
# from botocore.exceptions import ClientError
# from urllib.parse import urlparse
# import re
# import unicodedata
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# # Custom filter to sanitize non-ASCII characters for console
# class ConsoleSafeFilter(logging.Filter):
#     def filter(self, record):
#         if isinstance(record.msg, str):
#             record.msg = unicodedata.normalize('NFKD', record.msg).encode('ascii', 'replace').decode('ascii')
#         return True

# # Set up logging: console (ASCII-safe) and file (UTF-8)
# console_handler = logging.StreamHandler()
# console_handler.addFilter(ConsoleSafeFilter())
# console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# file_handler = logging.FileHandler('app.log', encoding='utf-8')
# file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# logging.basicConfig(
#     level=logging.INFO,
#     handlers=[console_handler, file_handler]
# )
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)
# app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback-secret-key-for-dev')

# # IBM COS Credentials
# COS_API_KEY = os.getenv('COS_API_KEY')
# COS_SERVICE_INSTANCE_ID = os.getenv('COS_SERVICE_INSTANCE_ID')
# COS_ENDPOINT = os.getenv('COS_ENDPOINT')
# COS_BUCKET = os.getenv('COS_BUCKET')

# # Assisto API endpoint
# ASSISTO_API_URL = os.getenv('ASSISTO_API_URL')

# # Watsonx API credentials
# api_key = os.getenv('WATSONX_API_KEY')
# url = os.getenv('WATSONX_URL')
# project_id = os.getenv('WATSONX_PROJECT_ID')
# model_id = os.getenv('WATSONX_MODEL_ID')
# auth_url = os.getenv('WATSONX_AUTH_URL',)

# # Initialize COS client with increased timeouts and retries
# try:
#     cos_client = ibm_boto3.client(
#         's3',
#         ibm_api_key_id=COS_API_KEY,
#         ibm_service_instance_id=COS_SERVICE_INSTANCE_ID,
#         config=Config(
#             signature_version='oauth',
#             connect_timeout=10,  # Increased from default 5s
#             read_timeout=30,     # Increased from default
#             retries={
#                 'max_attempts': 5,  # Increased from default 3
#                 'mode': 'standard'
#             }
#         ),
#         endpoint_url=COS_ENDPOINT
#     )
#     logger.info("Successfully initialized COS client")
# except Exception as e:
#     logger.error(f"Failed to initialize COS client: {e}")
#     raise

# # Global variable to store JSON files
# json_files = None

# def get_cos_files():
#     global json_files
#     if json_files is None:
#         start_time = time.time()
#         logger.info("Fetching and validating JSON files from IBM COS with multithreading")
#         valid_keys = []
#         try:
#             paginator = cos_client.get_paginator('list_objects_v2')
#             all_keys = []
#             for page in paginator.paginate(Bucket=COS_BUCKET):
#                 if 'Contents' not in page:
#                     logger.warning(f"No objects found in bucket: {COS_BUCKET}")
#                     break
#                 for obj in page.get('Contents', []):
#                     key = obj['Key']
#                     if key.endswith('.json'):
#                         all_keys.append(key)
#             logger.info(f"Found {len(all_keys)} JSON files to validate")

#             def validate_json(key):
#                 try:
#                     response = cos_client.get_object(Bucket=COS_BUCKET, Key=key)
#                     json_data = json.loads(response['Body'].read().decode('utf-8'))
#                     audio_url = json_data.get('AudioFile', '').strip()
#                     duration_str = json_data.get('CallDuration', '00:00:00')
#                     duration_seconds = parse_call_duration(duration_str)
#                     if audio_url and urllib.parse.urlparse(audio_url).scheme and duration_seconds > 15:
#                         logger.info(f"Validated {key} with valid URL {audio_url} and duration {duration_seconds}s")
#                         return key
#                     else:
#                         logger.warning(f"Skipping {key}: Invalid URL or duration <= 15s (duration: {duration_seconds}s)")
#                         return None
#                 except Exception as e:
#                     logger.error(f"Error validating {key}: {e}")
#                     return None

#             with ThreadPoolExecutor(max_workers=10) as executor:
#                 results = list(executor.map(validate_json, all_keys))
#                 json_files = [key for key in results if key is not None]
#             logger.info(f"Completed validation with {len(json_files)} valid JSON files in {time.time() - start_time:.2f} seconds")
#         except Exception as e:
#             logger.error(f"Failed to list or validate objects in COS: {e}")
#             json_files = []
#             return json_files  # Return empty list to allow app to continue
#     else:
#         logger.info(f"Reusing existing JSON files: {len(json_files)}")
#     return json_files

# def parse_call_duration(duration_str):
#     try:
#         time_obj = datetime.strptime(duration_str, "%H:%M:%S")
#         seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
#         logger.info(f"Parsed CallDuration '{duration_str}' to {seconds} seconds")
#         return seconds
#     except ValueError as e:
#         logger.error(f"Failed to parse CallDuration '{duration_str}': {e}")
#         return 0

# def access_token():
#     headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
#     data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
#     try:
#         # Use a session with increased timeouts and retries
#         session = requests.Session()
#         retries = Retry(
#             total=5,  # Increased from default
#             backoff_factor=2,  # Increased for exponential backoff
#             status_forcelist=[429, 500, 502, 503, 504]
#         )
#         session.mount('https://', HTTPAdapter(max_retries=retries))
#         response = session.post(auth_url, headers=headers, data=data, timeout=(10, 30))  # (connect, read)
#         response.raise_for_status()
#         return response.json()['access_token']
#     except requests.RequestException as e:
#         logger.error(f"Failed to get Watsonx access token: {e}")
#         raise

# def chunk_transcript(transcript, max_chunk_length=1000):
#     """Split transcript into chunks, fallback to character-based splitting if needed."""
#     try:
#         sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
#         chunks = []
#         current_chunk = []
#         current_length = 0

#         for sentence in sentences:
#             sentence_length = len(sentence)
#             if current_length + sentence_length > max_chunk_length:
#                 if current_chunk:
#                     chunks.append(' '.join(current_chunk))
#                     current_chunk = [sentence]
#                     current_length = sentence_length
#                 else:
#                     while sentence:
#                         chunks.append(sentence[:max_chunk_length])
#                         sentence = sentence[max_chunk_length:]
#                         current_length = 0
#             else:
#                 current_chunk.append(sentence)
#                 current_length += sentence_length

#         if current_chunk:
#             chunks.append(' '.join(current_chunk))

#         if not chunks:
#             chunks = [transcript[i:i+max_chunk_length] for i in range(0, len(transcript), max_chunk_length)]

#         logger.info(f"Split transcript into {len(chunks)} chunks: {[len(c) for c in chunks]}")
#         return chunks
#     except Exception as e:
#         logger.error(f"Failed to chunk transcript: {e}")
#         return [transcript[i:i+max_chunk_length] for i in range(0, len(transcript), max_chunk_length)]

# def create_watsonx_session():
#     """Create a requests session with retry logic and increased timeouts for Watsonx API."""
#     session = requests.Session()
#     retries = Retry(
#         total=5,  # Increased from 3
#         backoff_factor=2,  # Increased for exponential backoff
#         status_forcelist=[429, 500, 502, 503, 504],
#         allowed_methods=["POST"]
#     )
#     session.mount('https://', HTTPAdapter(max_retries=retries))
#     return session

# def process_chunk_separatespeakers(chunk, chunk_index, total_chunks):
#     """Process a single chunk for Separatespeakers."""
#     start_time = time.time()
#     prompt_template = """
#     Convert the following transcript into a conversation format with turns labeled as "Speaker1:" or "Speaker2:". The transcript may contain English and Hindi text. Alternate speakers naturally, ensuring each turn is on a new line.
#     Transcript:
#     {chunk}
#     Output format:
#     Speaker1: [Text]
#     Speaker2: [Text]
#     """
#     prompt = prompt_template.format(chunk=chunk)
#     body = {
#         "input": prompt,
#         "parameters": {
#             "decoding_method": "greedy",
#             "max_new_tokens": 1000,
#             "min_new_tokens": 30,
#             "repetition_penalty": 1.05,
#             "temperature": 0.1
#         },
#         "model_id": model_id,
#         "project_id": project_id
#     }
#     headers = {
#         "Accept": "application/json",
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {access_token()}"
#     }
#     try:
#         logger.info(f"Thread started for Separatespeakers chunk {chunk_index+1}/{total_chunks} with input length: {len(chunk)}")
#         with create_watsonx_session() as session:
#             response = session.post(url, headers=headers, json=body, timeout=(10, 30))
#             response.raise_for_status()
#             response_text = response.json()['results'][0]['generated_text']
#         logger.info(f"Thread completed for Separatespeakers chunk {chunk_index+1}/{total_chunks} in {time.time() - start_time:.2f}s")
#         with open('app.log', 'a', encoding='utf-8') as f:
#             f.write(f"{datetime.now()} - Separatespeakers chunk {chunk_index+1} response: {response_text}\n")

#         lines = response_text.split('\n')
#         clean_lines = []
#         for line in lines:
#             line = line.strip()
#             if not line or "transcript" in line.lower() or "conversation" in line.lower():
#                 continue
#             if line.startswith(('Speaker1:', 'Speaker2:', 'Speaker 1:', 'Speaker 2:', '**Speaker1**:', '**Speaker2**:', '**Speaker 1**:', '**Speaker 2**:')):
#                 line = line.replace('Speaker 1:', 'Speaker1:').replace('Speaker 2:', 'Speaker2:')
#                 line = line.replace('**Speaker1**:', 'Speaker1:').replace('**Speaker2**:', 'Speaker2:')
#                 line = line.replace('**Speaker 1**:', 'Speaker1:').replace('**Speaker 2**:', 'Speaker2:')
#                 clean_lines.append(line)
#         result = '\n'.join(clean_lines) if clean_lines else chunk
#         return chunk_index, result
#     except requests.RequestException as e:
#         logger.error(f"Failed to separate speakers for chunk {chunk_index+1}: {e}")
#         sentences = chunk.split('.')
#         alternate_lines = []
#         is_speaker1 = True
#         for sentence in sentences:
#             sentence = sentence.strip()
#             if sentence:
#                 speaker = "Speaker1" if is_speaker1 else "Speaker2"
#                 alternate_lines.append(f"{speaker}: {sentence}.")
#                 is_speaker1 = not is_speaker1
#         result = '\n'.join(alternate_lines)
#         return chunk_index, result

# def Separatespeakers(trans):
#     try:
#         chunks = chunk_transcript(trans, max_chunk_length=1000)
#         separated_chunks = [''] * len(chunks)

#         with ThreadPoolExecutor(max_workers=4) as executor:
#             futures = [
#                 executor.submit(process_chunk_separatespeakers, chunk, i, len(chunks))
#                 for i, chunk in enumerate(chunks)
#             ]
#             for future in futures:
#                 chunk_index, result = future.result()
#                 separated_chunks[chunk_index] = result

#         combined_transcript = '\n'.join(separated_chunks)
#         if combined_transcript.strip():
#             logger.info(f"Combined separated transcript length: {len(combined_transcript)}")
#             return combined_transcript
#         else:
#             logger.warning("No valid separated transcript produced, returning original")
#             return trans
#     except Exception as e:
#         logger.error(f"Unexpected error in Separatespeakers: {e}")
#         return trans

# def process_chunk_getcallquality(chunk, chunk_index, total_chunks):
#     """Process a single chunk for Getcallquality, filtering low-information content."""
#     start_time = time.time()
#     low_info_patterns = [
#         r'^(Speaker[12]: (ठीक है|धन्यवाद|अलविदा|sir|hello|hi|bye)\.? ?)+$',
#         r'^\s*$'
#     ]
#     if any(re.match(pattern, chunk.strip(), re.IGNORECASE) for pattern in low_info_patterns):
#         logger.info(f"Skipping low-information chunk {chunk_index+1}/{total_chunks}: {chunk[:50]}...")
#         return chunk_index, None

#     body = {
#         "input": f"""
#         Analyze the call quality of the following conversation transcript, which may contain English and Hindi text. Focus on meaningful interactions such as customer requests, agent responses, and issue resolution. Ignore repetitive greetings or farewells (e.g., "hello", "thank you", "goodbye", "ठीक है", "धन्यवाद", "अलविदा"). Provide a concise summary of the customer's request and the agent's action, a call rating, and the reason for the rating.
#         Output format:
#         Add-on Request by Customer: [Customer request]
#         Action Taken for the request: [Agent action]
#         Call Rating: [X out of 10]
#         Reason: [Reason for rating]
#         Transcript:
#         {chunk or 'No transcription available'}
#         """,
#         "parameters": {
#             "decoding_method": "greedy",
#             "max_new_tokens": 500,
#             "min_new_tokens": 50,
#             "stop_sequences": ["/"],
#             "repetition_penalty": 1.1,
#             "temperature": 0.3
#         },
#         "model_id": model_id,
#         "project_id": project_id
#     }
#     headers = {
#         "Accept": "application/json",
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {access_token()}"
#     }
#     try:
#         logger.info(f"Thread started for Getcallquality chunk {chunk_index+1}/{total_chunks} with input length: {len(chunk)}")
#         with create_watsonx_session() as session:
#             response = session.post(url, headers=headers, json=body, timeout=(10, 30))
#             response.raise_for_status()
#             result = response.json()['results'][0]['generated_text']
#         logger.info(f"Thread completed for Getcallquality chunk {chunk_index+1}/{total_chunks} in {time.time() - start_time:.2f}s")
#         with open('app.log', 'a', encoding='utf-8') as f:
#             f.write(f"{datetime.now()} - Getcallquality chunk {chunk_index+1} response: {result}\n")

#         request = "No request identified"
#         action = "No action identified"
#         rating = "Unknown"
#         reason = "No reason provided"
#         lines = result.split('\n')
#         for line in lines:
#             line = line.strip()
#             if line.startswith("Add-on Request by Customer:"):
#                 request = line[len("Add-on Request by Customer:"):].strip()
#             elif line.startswith("Action Taken for the request:"):
#                 action = line[len("Action Taken for the request:"):].strip()
#             elif line.startswith("Call Rating:"):
#                 rating = line[len("Call Rating:"):].strip()
#             elif line.startswith("Reason:"):
#                 reason = line[len("Reason:"):].strip()

#         return chunk_index, {
#             "request": request,
#             "action": action,
#             "rating": rating,
#             "reason": reason
#         }
#     except requests.RequestException as e:
#         logger.error(f"Failed to get call quality for chunk {chunk_index+1}: {e}")
#         return chunk_index, None

# def Getcallquality(trans):
#     chunks = chunk_transcript(trans, max_chunk_length=1000)
#     quality_results = [None] * len(chunks)

#     with ThreadPoolExecutor(max_workers=4) as executor:
#         futures = [
#             executor.submit(process_chunk_getcallquality, chunk, i, len(chunks))
#             for i, chunk in enumerate(chunks)
#         ]
#         for future in futures:
#             chunk_index, result = future.result()
#             quality_results[chunk_index] = result

#     valid_results = [r for r in quality_results if r is not None]
#     if not valid_results:
#         return "Error: No meaningful call quality results"

#     requests = []
#     actions = []
#     reasons = []
#     ratings = []
#     for result in valid_results:
#         if result["request"] != "No request identified":
#             requests.append(result["request"])
#         if result["action"] != "No action identified":
#             actions.append(result["action"])
#         if result["reason"] != "No reason provided":
#             reasons.append(result["reason"])
#         if result["rating"] != "Unknown" and "out of" in result["rating"]:
#             try:
#                 rating_value = float(result["rating"].split(" out of ")[0])
#                 ratings.append(rating_value)
#             except (ValueError, IndexError):
#                 pass

#     combined_request = " ".join(requests) or "No specific customer request identified"
#     combined_action = " ".join(actions) or "No specific action taken by agent"
#     combined_reason = " ".join(reasons) or "No specific reason provided"
#     combined_rating = f"{round(sum(ratings) / len(ratings), 1) if ratings else 'Unknown'} out of 10"

#     return f"""Add-on Request by Customer: {combined_request}
# Action Taken for the request: {combined_action}
# Call Rating: {combined_rating}
# Reason: {combined_reason}"""

# def process_chunk_getinsights(chunk, chunk_index, total_chunks):
#     """Process a single chunk for Getinsights."""
#     start_time = time.time()
#     body = {
#         "input": f"""
#         Provide an insight summary and sentiment for the following conversation transcript, which may contain English and Hindi text.
#         Output format:
#         Insights Summary: insight summary...
#         Sentiment: sentiment...
#         Transcript:
#         {chunk or 'No transcription available'}
#         """,
#         "parameters": {
#             "decoding_method": "greedy",
#             "max_new_tokens": 1000,
#             "min_new_tokens": 30,
#             "stop_sequences": ["/"],
#             "repetition_penalty": 1.05,
#             "temperature": 0.5
#         },
#         "model_id": model_id,
#         "project_id": project_id
#     }
#     headers = {
#         "Accept": "application/json",
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {access_token()}"
#     }
#     try:
#         logger.info(f"Thread started for Getinsights chunk {chunk_index+1}/{total_chunks} with input length: {len(chunk)}")
#         with create_watsonx_session() as session:
#             response = session.post(url, headers=headers, json=body, timeout=(10, 30))
#             response.raise_for_status()
#             generated_text = response.json()['results'][0]['generated_text']
#         logger.info(f"Thread completed for Getinsights chunk {chunk_index+1}/{total_chunks} in {time.time() - start_time:.2f}s")
#         with open('app.log', 'a', encoding='utf-8') as f:
#             f.write(f"{datetime.now()} - Getinsights chunk {chunk_index+1} response: {generated_text}\n")

#         insight_summary = "No summary available"
#         sentiment = "Unknown"
#         lines = generated_text.split('\n')
#         for line in lines:
#             line = line.strip()
#             if line.startswith("Insights Summary:"):
#                 insight_summary = line[len("Insights Summary:"):].strip()
#             elif line.startswith("Sentiment:"):
#                 sentiment = line[len("Sentiment:"):].strip()
#         return chunk_index, (insight_summary, sentiment)
#     except requests.RequestException as e:
#         logger.error(f"Failed to get insights for chunk {chunk_index+1}: {e}")
#         return chunk_index, ("Error: Unable to retrieve insights", "Unknown")

# def Getinsights(trans):
#     chunks = chunk_transcript(trans, max_chunk_length=1000)
#     results = [None] * len(chunks)

#     with ThreadPoolExecutor(max_workers=4) as executor:
#         futures = [
#             executor.submit(process_chunk_getinsights, chunk, i, len(chunks))
#             for i, chunk in enumerate(chunks)
#         ]
#         for future in futures:
#             chunk_index, result = future.result()
#             results[chunk_index] = result

#     summaries = [r[0] for r in results if r]
#     sentiments = [r[1] for r in results if r]
#     combined_summary = " ".join([s for s in summaries if s and "Error" not in s]) or "No insights available"
#     combined_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "Unknown"
#     return [combined_summary, combined_sentiment]

# def create_json_output(file_key, transcript, insights, callquality, separated):
#     logger.info(f"Processing insights: {insights!r}")
#     try:
#         insight_summary, sentiment = insights
#     except ValueError as e:
#         logger.error(f"Error unpacking insights: {e}. Insights: {insights!r}")
#         insight_summary = "Error: Could not extract summary"
#         sentiment = "Error: Could not extract sentiment"

#     output_data = {
#         "call_transcription": {
#             "raw_transcript": transcript,
#             "separated_transcript": separated
#         },
#         "call_insight": {
#             "summary": insight_summary
#         },
#         "call_rating": {
#             "quality": callquality
#         },
#         "call_sentiment_analysis": {
#             "sentiment": sentiment
#         }
#     }
#     try:
#         json_str = json.dumps(output_data, indent=2, ensure_ascii=False)
#         filename = f"output_{file_key.replace('.json', '')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
#         os.makedirs('temp', exist_ok=True)
#         temp_path = os.path.join('temp', filename)
#         with open(temp_path, 'w', encoding='utf-8') as f:
#             f.write(json_str)
#         session['json_filename'] = filename
#         session['json_path'] = temp_path
#         logger.info(f"Created JSON output: {temp_path}")
#         return json_str
#     except Exception as e:
#         logger.error(f"Failed to create JSON output: {e}")
#         return json.dumps({"error": "Failed to create JSON output"})

# def get_json_from_cos(bucket, key):
#     try:
#         response = cos_client.get_object(Bucket=bucket, Key=key)
#         json_data = json.loads(response['Body'].read().decode('utf-8'))
#         logger.info(f"Retrieved JSON data for {key}")
#         return json_data
#     except ClientError as e:
#         logger.error(f"Failed to retrieve JSON from COS: {e}")
#         return None
#     except json.JSONDecodeError as e:
#         logger.error(f"Invalid JSON format for key {key}: {e}")
#         return None

# def download_audio_file(audio_url):
#     try:
#         logger.info(f"Downloading audio from: {audio_url}")
#         audio_response = requests.get(audio_url, stream=True, timeout=(10, 30))
#         audio_response.raise_for_status()
#         parsed_url = urlparse(audio_url)
#         filename = os.path.basename(parsed_url.path) or f"temp_audio_{time.time()}.mp3"
#         with open(filename, 'wb') as f:
#             for chunk in audio_response.iter_content(chunk_size=8192):
#                 if chunk:
#                     f.write(chunk)
#         logger.info(f"Successfully downloaded audio to: {filename}")
#         return filename
#     except requests.RequestException as e:
#         logger.error(f"Error downloading audio file from {audio_url}: {e}")
#         return None

# def process_audio_with_assisto(filename, apikey=None, json_data=None):
#     try:
#         if not filename or not os.path.exists(filename):
#             logger.error(f"File not found: {filename}")
#             return {"status": "error", "response": f"File not found: {filename}"}
        
#         with open(filename, 'rb') as f:
#             files = [('file', (filename, f, 'audio/mpeg'))]
#             headers = {}
#             if apikey:
#                 headers['Authorization'] = f"Bearer {apikey}"
            
#             payload = {
#                 "transcribe": "true",
#                 "monitorUCID": json_data.get("monitorUCID", ""),
#                 "AgentID": json_data.get("AgentID", "")
#             }
            
#             logger.info(f"Sending request to Assisto API with payload: {payload}")
#             session = create_watsonx_session()  # Reuse session with retries
#             response = session.post(ASSISTO_API_URL, headers=headers, data=payload, files=files, timeout=(10, 30))
#             response.raise_for_status()
            
#             if response.status_code == 200:
#                 raw_response = response.text
#                 logger.info(f"Raw Assisto API response length: {len(raw_response)}")
                
#                 try:
#                     response_data = response.json()
#                     if isinstance(response_data, dict) and 'result' in response_data and response_data['result']:
#                         speaker_texts = {}
#                         for item in response_data['result']:
#                             speaker = item.get('speaker', '0')
#                             message = item.get('message', '').strip()
#                             if message:
#                                 if speaker in speaker_texts:
#                                     speaker_texts[speaker] += ' ' + message
#                                 else:
#                                     speaker_texts[speaker] = message
#                         combined_transcript = ' '.join(speaker_texts.values()) or json.dumps(response_data)
#                         logger.info(f"Combined transcript from Assisto length: {len(combined_transcript)}")
#                         return {"status": "success", "transcription": combined_transcript, "response": response_data}
#                     else:
#                         logger.warning(f"Unexpected response format from Assisto: {response_data}")
#                         return {"status": "success", "transcription": raw_response, "response": response_data}
#                 except ValueError:
#                     if len(response.text.strip()) > 50:
#                         logger.warning(f"Non-JSON response from Assisto: {response.text[:200]}...")
#                         return {"status": "success", "transcription": response.text, "response": response.text}
#                     return {"status": "success", "transcription": None, "response": response.text}
#             else:
#                 logger.error(f"Assisto API failed with status {response.status_code}: {response.text}")
#                 return {"status": "error", "response": f"API request failed with status code {response.status_code}"}
#     except requests.RequestException as e:
#         logger.error(f"Error during Assisto API request: {e}")
#         return {"status": "error", "response": f"Error during API request: {e}"}
#     except FileNotFoundError:
#         logger.error(f"File not found: {filename}")
#         return {"status": "error", "response": f"File not found: {filename}"}
#     except Exception as e:
#         logger.error(f"Unexpected error in process_audio_with_assisto: {e}")
#         return {"status": "error", "response": f"Unexpected error: {e}"}

# def process_audio_from_cos(file_key):
#     errors = []
#     session.clear()
#     try:
#         logger.info(f"Processing file from COS: {file_key}")
#         json_data = get_json_from_cos(COS_BUCKET, file_key)
#         if not json_data:
#             errors.append("Failed to retrieve or parse JSON data.")
#             return errors, None, None, None, None
        
#         audio_url = json_data.get('AudioFile', '').strip()
#         apikey = json_data.get('Apikey')
        
#         if not audio_url or not urllib.parse.urlparse(audio_url).scheme:
#             logger.error(f"Skipping {file_key} due to invalid AudioFile URL: {audio_url}")
#             errors.append(f"Invalid or missing AudioFile URL in {file_key}")
#             return errors, None, None, None, None
        
#         logger.info(f"Downloading audio from {audio_url}")
#         filename = download_audio_file(audio_url)
#         if not filename:
#             errors.append("Failed to download audio file.")
#             return errors, None, None, None, None
        
#         logger.info(f"Processing audio file {filename} with Assisto API")
#         assisto_result = process_audio_with_assisto(filename, apikey, json_data)
#         if assisto_result["status"] == "error":
#             errors.append(f"Assisto API error: {assisto_result['response']}")
#             return errors, None, None, None, None
#         transcript = assisto_result.get("transcription", "No transcription available.")
#         logger.info(f"Assisto transcription length: {len(transcript)}")
        
#         try:
#             os.remove(filename)
#             logger.info(f"Temporary file {filename} deleted.")
#         except OSError as e:
#             logger.error(f"Failed to delete temporary file: {e}")
        
#         separated_transcript = Separatespeakers(transcript)
#         logger.info(f"Separated transcript length: {len(separated_transcript)}")
        
#         insights = Getinsights(separated_transcript)
#         callquality = Getcallquality(separated_transcript)
#         logger.info(f"Insights: {insights}")
#         logger.info(f"Call quality: {callquality}")
        
#         create_json_output(file_key, transcript, insights, callquality, separated_transcript)
#         return errors, separated_transcript, insights, callquality, separated_transcript
    
#     except Exception as e:
#         logger.error(f"Unexpected error in process_audio_from_cos: {e}")
#         errors.append(f"Error processing {file_key}: {e}")
#         return errors, None, None, None, None

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     session.setdefault('transcript', "Select a JSON file to get call transcript")
#     session.setdefault('insights', "Select a JSON file to get insights")
#     session.setdefault('callquality', "Select a JSON file to get call quality")
#     session.setdefault('separated', "Select a JSON file to get separated transcript")
#     session.setdefault('json_filename', None)
#     session.setdefault('json_path', None)

#     try:
#         json_files = get_cos_files()
#     except Exception as e:
#         logger.error(f"Error fetching COS files: {e}")
#         json_files = []

#     errors = []
#     if request.method == 'POST':
#         selected_file = request.form.get('selected_file')
#         if selected_file and selected_file in json_files:
#             logger.info(f"Processing selected file: {selected_file}")
#             errors, separated, insights, callquality, _ = process_audio_from_cos(selected_file)
#             if not errors and separated and separated != "No valid transcription available.":
#                 session['separated'] = separated
#                 if isinstance(insights, list) and len(insights) == 2:
#                     session['insights'] = f"Insights Summary: {insights[0]}\nSentiment: {insights[1]}"
#                 else:
#                     session['insights'] = "No insights available."
#                 session['callquality'] = callquality or "No call quality available."
#                 logger.info(f"Updated session - separated transcript length: {len(session['separated'])}")
#             else:
#                 session['separated'] = f"Error: {' '.join(errors) if errors else 'No valid transcription available.'}"
#                 logger.error(f"Errors or no valid transcript: {errors or 'None'}")
#             session['last_processed'] = selected_file
#             return redirect(url_for('index'))
#         else:
#             errors.append("Invalid file selection or file not found.")

#     if 'last_processed' in session and session['last_processed'] in json_files:
#         selected_file = session['last_processed']
#     else:
#         selected_file = None

#     return render_template('index.html', 
#                          json_files=json_files, 
#                          call_transcript=session['separated'],
#                          call_insights=session['insights'],
#                          call_quality=session['callquality'],
#                          failed_files=[], 
#                          errors=errors,
#                          json_filename=session['json_filename'],
#                          selected_file=selected_file)

# @app.route('/api/process', methods=['POST'])
# def api_process():
#     try:
#         data = request.get_json()
#         if not data or 'file_key' not in data:
#             return jsonify({"error": "Missing file_key in request"}), 400
#         file_key = data['file_key']
#         json_files = get_cos_files()
#         if file_key not in json_files:
#             return jsonify({"error": f"Invalid file_key: {file_key} not found in {len(json_files)} files"}), 400
#         logger.info(f"Processing API request for file: {file_key}")
#         errors, separated, insights, callquality, transcript = process_audio_from_cos(file_key)
#         if errors:
#             return jsonify({"error": errors[0], "status": "failed"}), 400
#         insight_summary, sentiment = insights
#         response = {
#             "data": {
#                 "call_transcription": {
#                     "raw_transcript": transcript,
#                     "separated_transcript": separated
#                 },
#                 "call_insight": {
#                     "summary": insight_summary
#                 },
#                 "call_rating": {
#                     "quality": callquality
#                 },
#                 "call_sentiment_analysis": {
#                     "sentiment": sentiment
#                 }
#             }
#         }
#         logger.info(f"API response: {response}")
#         return jsonify(response), 200
#     except Exception as e:
#         logger.error(f"Exception in api_process: {str(e)}", exc_info=True)
#         return jsonify({"error": "Internal server error", "details": str(e)}), 500

# @app.route('/download_json')
# def download_json():
#     try:
#         if 'json_path' in session and os.path.exists(session['json_path']):
#             return send_file(session['json_path'], as_attachment=True, download_name=session['json_filename'])
#         logger.warning("No JSON file available for download")
#         return redirect(url_for('index'))
#     except Exception as e:
#         logger.error(f"Error downloading JSON: {e}")
#         return redirect(url_for('index'))

# if __name__ == '__main__':
#     try:
#         port = int(os.getenv('PORT', 8000))
#         logger.info(f"Starting Flask app on port {port}")
#         app.run(debug=False, host='0.0.0.0', port=port)
#     except Exception as e:
#         logger.error(f"Failed to start Flask app: {e}")
#         raise/


from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify
import requests
import json
import ibm_boto3
from ibm_botocore.client import Config
from ibm_botocore.config import Config as BotocoreConfig
import urllib.parse
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from botocore.exceptions import ClientError
from urllib.parse import urlparse
import re
import unicodedata
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Custom filter to sanitize non-ASCII characters for console
class ConsoleSafeFilter(logging.Filter):
    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = unicodedata.normalize('NFKD', record.msg).encode('ascii', 'replace').decode('ascii')
        return True

# Set up logging: console (ASCII-safe) and file (UTF-8)
console_handler = logging.StreamHandler()
console_handler.addFilter(ConsoleSafeFilter())
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback-secret-key-for-dev')

# IBM COS Credentials
COS_API_KEY = os.getenv('COS_API_KEY')
COS_SERVICE_INSTANCE_ID = os.getenv('COS_SERVICE_INSTANCE_ID')
COS_ENDPOINT = os.getenv('COS_ENDPOINT')
COS_BUCKET = os.getenv('COS_BUCKET')

# Assisto API endpoint
ASSISTO_API_URL = os.getenv('ASSISTO_API_URL')

# Watsonx API credentials
api_key = os.getenv('WATSONX_API_KEY')
url = os.getenv('WATSONX_URL')
project_id = os.getenv('WATSONX_PROJECT_ID')
model_id = os.getenv('WATSONX_MODEL_ID')
auth_url = os.getenv('WATSONX_AUTH_URL')

# Initialize COS client with increased timeouts and retries
try:
    cos_client = ibm_boto3.client(
        's3',
        ibm_api_key_id=COS_API_KEY,
        ibm_service_instance_id=COS_SERVICE_INSTANCE_ID,
        config=Config(
            signature_version='oauth',
            connect_timeout=30,  # Max connect timeout
            read_timeout=300,    # Max read timeout (5 minutes)
            retries={
                'max_attempts': 10,  # Max retries
                'mode': 'standard'
            }
        ),
        endpoint_url=COS_ENDPOINT
    )
    logger.info("Successfully initialized COS client")
except Exception as e:
    logger.error(f"Failed to initialize COS client: {e}")
    raise

# Global variable to store JSON files
json_files = None

def get_cos_files():
    global json_files
    if json_files is None:
        start_time = time.time()
        logger.info("Fetching and validating JSON files from IBM COS with multithreading")
        valid_keys = []
        try:
            paginator = cos_client.get_paginator('list_objects_v2')
            all_keys = []
            for page in paginator.paginate(Bucket=COS_BUCKET):
                if 'Contents' not in page:
                    logger.warning(f"No objects found in bucket: {COS_BUCKET}")
                    break
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('.json'):
                        all_keys.append(key)
            logger.info(f"Found {len(all_keys)} JSON files to validate")

            def validate_json(key):
                try:
                    response = cos_client.get_object(Bucket=COS_BUCKET, Key=key)
                    json_data = json.loads(response['Body'].read().decode('utf-8'))
                    audio_url = json_data.get('AudioFile', '').strip()
                    duration_str = json_data.get('CallDuration', '00:00:00')
                    duration_seconds = parse_call_duration(duration_str)
                    if audio_url and urllib.parse.urlparse(audio_url).scheme and duration_seconds > 15:
                        logger.info(f"Validated {key} with valid URL {audio_url} and duration {duration_seconds}s")
                        return key
                    else:
                        logger.warning(f"Skipping {key}: Invalid URL or duration <= 15s (duration: {duration_seconds}s)")
                        return None
                except Exception as e:
                    logger.error(f"Error validating {key}: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(validate_json, all_keys))
                json_files = [key for key in results if key is not None]
            logger.info(f"Completed validation with {len(json_files)} valid JSON files in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to list or validate objects in COS: {e}")
            json_files = []
            return json_files
    else:
        logger.info(f"Reusing existing JSON files: {len(json_files)}")
    return json_files

def parse_call_duration(duration_str):
    try:
        time_obj = datetime.strptime(duration_str, "%H:%M:%S")
        seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
        logger.info(f"Parsed CallDuration '{duration_str}' to {seconds} seconds")
        return seconds
    except ValueError as e:
        logger.error(f"Failed to parse CallDuration '{duration_str}': {e}")
        return 0

def access_token():
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": api_key}
    try:
        session = requests.Session()
        retries = Retry(
            total=10,  # Max retries
            backoff_factor=5,  # Max backoff
            status_forcelist=[429, 500, 502, 503, 504]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.post(auth_url, headers=headers, data=data, timeout=(30, 300))  # Max timeouts
        response.raise_for_status()
        return response.json()['access_token']
    except requests.RequestException as e:
        logger.error(f"Failed to get Watsonx access token: {e}")
        raise

def chunk_transcript(transcript, max_chunk_length=800):  # Reduced for optimization
    try:
        sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > max_chunk_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    while sentence:
                        chunks.append(sentence[:max_chunk_length])
                        sentence = sentence[max_chunk_length:]
                        current_length = 0
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        if not chunks:
            chunks = [transcript[i:i+max_chunk_length] for i in range(0, len(transcript), max_chunk_length)]

        logger.info(f"Split transcript into {len(chunks)} chunks: {[len(c) for c in chunks]}")
        return chunks
    except Exception as e:
        logger.error(f"Failed to chunk transcript: {e}")
        return [transcript[i:i+max_chunk_length] for i in range(0, len(transcript), max_chunk_length)]

def create_watsonx_session():
    session = requests.Session()
    retries = Retry(
        total=10,  # Max retries
        backoff_factor=5,  # Max backoff
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def process_chunk_separatespeakers(chunk, chunk_index, total_chunks):
    start_time = time.time()
    prompt_template = """
    Convert the following transcript into a conversation format with turns labeled as "Speaker1:" or "Speaker2:". The transcript may contain English and Hindi text. Alternate speakers naturally, ensuring each turn is on a new line.
    Transcript:
    {chunk}
    Output format:
    Speaker1: [Text]
    Speaker2: [Text]
    """
    prompt = prompt_template.format(chunk=chunk)
    body = {
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 1000,
            "min_new_tokens": 30,
            "repetition_penalty": 1.05,
            "temperature": 0.1
        },
        "model_id": model_id,
        "project_id": project_id
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token()}"
    }
    try:
        logger.info(f"Thread started for Separatespeakers chunk {chunk_index+1}/{total_chunks} with input length: {len(chunk)}")
        with create_watsonx_session() as session:
            response = session.post(url, headers=headers, json=body, timeout=(30, 300))  # Max timeouts
            response.raise_for_status()
            response_text = response.json()['results'][0]['generated_text']
        logger.info(f"Thread completed for Separatespeakers chunk {chunk_index+1}/{total_chunks} in {time.time() - start_time:.2f}s")
        with open('app.log', 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now()} - Separatespeakers chunk {chunk_index+1} response: {response_text}\n")

        lines = response_text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line or "transcript" in line.lower() or "conversation" in line.lower():
                continue
            if line.startswith(('Speaker1:', 'Speaker2:', 'Speaker 1:', 'Speaker 2:', '**Speaker1**:', '**Speaker2**:', '**Speaker 1**:', '**Speaker 2**:')):
                line = line.replace('Speaker 1:', 'Speaker1:').replace('Speaker 2:', 'Speaker2:')
                line = line.replace('**Speaker1**:', 'Speaker1:').replace('**Speaker2**:', 'Speaker2:')
                line = line.replace('**Speaker 1**:', 'Speaker1:').replace('**Speaker 2**:', 'Speaker2:')
                clean_lines.append(line)
        result = '\n'.join(clean_lines) if clean_lines else chunk
        return chunk_index, result
    except requests.RequestException as e:
        logger.error(f"Failed to separate speakers for chunk {chunk_index+1}: {e}")
        sentences = chunk.split('.')
        alternate_lines = []
        is_speaker1 = True
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                speaker = "Speaker1" if is_speaker1 else "Speaker2"
                alternate_lines.append(f"{speaker}: {sentence}.")
                is_speaker1 = not is_speaker1
        result = '\n'.join(alternate_lines)
        return chunk_index, result

def Separatespeakers(trans):
    try:
        chunks = chunk_transcript(trans, max_chunk_length=800)
        separated_chunks = [''] * len(chunks)

        with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers
            futures = [
                executor.submit(process_chunk_separatespeakers, chunk, i, len(chunks))
                for i, chunk in enumerate(chunks)
            ]
            for future in futures:
                chunk_index, result = future.result()
                separated_chunks[chunk_index] = result
                time.sleep(1)  # Delay to prevent API overload

        combined_transcript = '\n'.join(separated_chunks)
        if combined_transcript.strip():
            logger.info(f"Combined separated transcript length: {len(combined_transcript)}")
            return combined_transcript
        else:
            logger.warning("No valid separated transcript produced, returning original")
            return trans
    except Exception as e:
        logger.error(f"Unexpected error in Separatespeakers: {e}")
        return trans

def process_chunk_getcallquality(chunk, chunk_index, total_chunks):
    start_time = time.time()
    low_info_patterns = [
        r'^(Speaker[12]: (ठीक है|धन्यवाद|अलविदा|sir|hello|hi|bye)\.? ?)+$',
        r'^\s*$'
    ]
    if any(re.match(pattern, chunk.strip(), re.IGNORECASE) for pattern in low_info_patterns):
        logger.info(f"Skipping low-information chunk {chunk_index+1}/{total_chunks}: {chunk[:50]}...")
        return chunk_index, None

    body = {
        "input": f"""
        Analyze the call quality of the following conversation transcript, which may contain English and Hindi text. Focus on meaningful interactions such as customer requests, agent responses, and issue resolution. Ignore repetitive greetings or farewells (e.g., "hello", "thank you", "goodbye", "ठीक है", "धन्यवाद", "अलविदा"). Provide a concise summary of the customer's request and the agent's action, a call rating, and the reason for the rating.
        Output format:
        Add-on Request by Customer: [Customer request]
        Action Taken for the request: [Agent action]
        Call Rating: [X out of 10]
        Reason: [Reason for rating]
        Transcript:
        {chunk or 'No transcription available'}
        """,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 500,
            "min_new_tokens": 50,
            "stop_sequences": ["/"],
            "repetition_penalty": 1.1,
            "temperature": 0.3
        },
        "model_id": model_id,
        "project_id": project_id
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token()}"
    }
    try:
        logger.info(f"Thread started for Getcallquality chunk {chunk_index+1}/{total_chunks} with input length: {len(chunk)}")
        with create_watsonx_session() as session:
            response = session.post(url, headers=headers, json=body, timeout=(30, 300))  # Max timeouts
            response.raise_for_status()
            result = response.json()['results'][0]['generated_text']
        logger.info(f"Thread completed for Getcallquality chunk {chunk_index+1}/{total_chunks} in {time.time() - start_time:.2f}s")
        with open('app.log', 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now()} - Getcallquality chunk {chunk_index+1} response: {result}\n")

        request = "No request identified"
        action = "No action identified"
        rating = "Unknown"
        reason = "No reason provided"
        lines = result.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Add-on Request by Customer:"):
                request = line[len("Add-on Request by Customer:"):].strip()
            elif line.startswith("Action Taken for the request:"):
                action = line[len("Action Taken for the request:"):].strip()
            elif line.startswith("Call Rating:"):
                rating = line[len("Call Rating:"):].strip()
            elif line.startswith("Reason:"):
                reason = line[len("Reason:"):].strip()

        return chunk_index, {
            "request": request,
            "action": action,
            "rating": rating,
            "reason": reason
        }
    except requests.RequestException as e:
        logger.error(f"Failed to get call quality for chunk {chunk_index+1}: {e}")
        return chunk_index, None

def Getcallquality(trans):
    chunks = chunk_transcript(trans, max_chunk_length=800)
    quality_results = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_chunk_getcallquality, chunk, i, len(chunks))
            for i, chunk in enumerate(chunks)
        ]
        for future in futures:
            chunk_index, result = future.result()
            quality_results[chunk_index] = result
            time.sleep(1)  # Delay to prevent API overload

    valid_results = [r for r in quality_results if r is not None]
    if not valid_results:
        return "Error: No meaningful call quality results"

    requests = []
    actions = []
    reasons = []
    ratings = []
    for result in valid_results:
        if result["request"] != "No request identified":
            requests.append(result["request"])
        if result["action"] != "No action identified":
            actions.append(result["action"])
        if result["reason"] != "No reason provided":
            reasons.append(result["reason"])
        if result["rating"] != "Unknown" and "out of" in result["rating"]:
            try:
                rating_value = float(result["rating"].split(" out of ")[0])
                ratings.append(rating_value)
            except (ValueError, IndexError):
                pass

    combined_request = " ".join(requests) or "No specific customer request identified"
    combined_action = " ".join(actions) or "No specific action taken by agent"
    combined_reason = " ".join(reasons) or "No specific reason provided"
    combined_rating = f"{round(sum(ratings) / len(ratings), 1) if ratings else 'Unknown'} out of 10"

    return f"""Add-on Request by Customer: {combined_request}
Action Taken for the request: {combined_action}
Call Rating: {combined_rating}
Reason: {combined_reason}"""

def process_chunk_getinsights(chunk, chunk_index, total_chunks):
    start_time = time.time()
    body = {
        "input": f"""
        Provide an insight summary and sentiment for the following conversation transcript, which may contain English and Hindi text.
        Output format:
        Insights Summary: insight summary...
        Sentiment: sentiment...
        Transcript:
        {chunk or 'No transcription available'}
        """,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 1000,
            "min_new_tokens": 30,
            "stop_sequences": ["/"],
            "repetition_penalty": 1.05,
            "temperature": 0.5
        },
        "model_id": model_id,
        "project_id": project_id
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token()}"
    }
    try:
        logger.info(f"Thread started for Getinsights chunk {chunk_index+1}/{total_chunks} with input length: {len(chunk)}")
        with create_watsonx_session() as session:
            response = session.post(url, headers=headers, json=body, timeout=(30, 300))  # Max timeouts
            response.raise_for_status()
            generated_text = response.json()['results'][0]['generated_text']
        logger.info(f"Thread completed for Getinsights chunk {chunk_index+1}/{total_chunks} in {time.time() - start_time:.2f}s")
        with open('app.log', 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now()} - Getinsights chunk {chunk_index+1} response: {generated_text}\n")

        insight_summary = "No summary available"
        sentiment = "Unknown"
        lines = generated_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("Insights Summary:"):
                insight_summary = line[len("Insights Summary:"):].strip()
            elif line.startswith("Sentiment:"):
                sentiment = line[len("Sentiment:"):].strip()
        return chunk_index, (insight_summary, sentiment)
    except requests.RequestException as e:
        logger.error(f"Failed to get insights for chunk {chunk_index+1}: {e}")
        if len(chunk) > 400:
            logger.info(f"Retrying chunk {chunk_index+1} with smaller size")
            smaller_chunks = chunk_transcript(chunk, max_chunk_length=400)
            for i, small_chunk in enumerate(smaller_chunks):
                try:
                    logger.info(f"Retrying sub-chunk {chunk_index+1}.{i+1} with length {len(small_chunk)}")
                    with create_watsonx_session() as session:
                        response = session.post(url, headers=headers, json=body, timeout=(30, 300))
                        response.raise_for_status()
                        generated_text = response.json()['results'][0]['generated_text']
                    insight_summary = "Partial summary from retry"
                    sentiment = "Unknown"
                    for line in generated_text.split('\n'):
                        line = line.strip()
                        if line.startswith("Insights Summary:"):
                            insight_summary = line[len("Insights Summary:"):].strip()
                        elif line.startswith("Sentiment:"):
                            sentiment = line[len("Sentiment:"):].strip()
                    return chunk_index, (insight_summary, sentiment)
                except requests.RequestException as e2:
                    logger.error(f"Failed retry for sub-chunk {chunk_index+1}.{i+1}: {e2}")
                    continue
        return chunk_index, ("Error: Unable to retrieve insights", "Unknown")

def Getinsights(trans):
    chunks = chunk_transcript(trans, max_chunk_length=800)
    results = [None] * len(chunks)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(process_chunk_getinsights, chunk, i, len(chunks))
            for i, chunk in enumerate(chunks)
        ]
        for future in futures:
            chunk_index, result = future.result()
            results[chunk_index] = result
            time.sleep(1)  # Delay to prevent API overload

    summaries = [r[0] for r in results if r]
    sentiments = [r[1] for r in results if r]
    combined_summary = " ".join([s for s in summaries if s and "Error" not in s]) or "No insights available"
    combined_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "Unknown"
    return [combined_summary, combined_sentiment]

def create_json_output(file_key, transcript, insights, callquality, separated):
    logger.info(f"Processing insights: {insights!r}")
    try:
        insight_summary, sentiment = insights
    except ValueError as e:
        logger.error(f"Error unpacking insights: {e}. Insights: {insights!r}")
        insight_summary = "Error: Could not extract summary"
        sentiment = "Error: Could not extract sentiment"

    output_data = {
        "call_transcription": {
            "raw_transcript": transcript,
            "separated_transcript": separated
        },
        "call_insight": {
            "summary": insight_summary
        },
        "call_rating": {
            "quality": callquality
        },
        "call_sentiment_analysis": {
            "sentiment": sentiment
        }
    }
    try:
        json_str = json.dumps(output_data, indent=2, ensure_ascii=False)
        filename = f"output_{file_key.replace('.json', '')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        os.makedirs('temp', exist_ok=True)
        temp_path = os.path.join('temp', filename)
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        session['json_filename'] = filename
        session['json_path'] = temp_path
        logger.info(f"Created JSON output: {temp_path}")
        return json_str
    except Exception as e:
        logger.error(f"Failed to create JSON output: {e}")
        return json.dumps({"error": "Failed to create JSON output"})

def get_json_from_cos(bucket, key):
    try:
        response = cos_client.get_object(Bucket=bucket, Key=key)
        json_data = json.loads(response['Body'].read().decode('utf-8'))
        logger.info(f"Retrieved JSON data for {key}")
        return json_data
    except ClientError as e:
        logger.error(f"Failed to retrieve JSON from COS: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format for key {key}: {e}")
        return None

def download_audio_file(audio_url):
    try:
        logger.info(f"Downloading audio from: {audio_url}")
        audio_response = requests.get(audio_url, stream=True, timeout=(30, 300))  # Max timeouts
        audio_response.raise_for_status()
        parsed_url = urlparse(audio_url)
        filename = os.path.basename(parsed_url.path) or f"temp_audio_{time.time()}.mp3"
        with open(filename, 'wb') as f:
            for chunk in audio_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Successfully downloaded audio to: {filename}")
        return filename
    except requests.RequestException as e:
        logger.error(f"Error downloading audio file from {audio_url}: {e}")
        return None

def process_audio_with_assisto(filename, apikey=None, json_data=None):
    try:
        if not filename or not os.path.exists(filename):
            logger.error(f"File not found: {filename}")
            return {"status": "error", "response": f"File not found: {filename}"}
        
        with open(filename, 'rb') as f:
            files = [('file', (filename, f, 'audio/mpeg'))]
            headers = {}
            if apikey:
                headers['Authorization'] = f"Bearer {apikey}"
            
            payload = {
                "transcribe": "true",
                "monitorUCID": json_data.get("monitorUCID", ""),
                "AgentID": json_data.get("AgentID", "")
            }
            
            logger.info(f"Sending request to Assisto API with payload: {payload}")
            session = create_watsonx_session()
            response = session.post(ASSISTO_API_URL, headers=headers, data=payload, files=files, timeout=(30, 300))  # Max timeouts
            response.raise_for_status()
            
            if response.status_code == 200:
                raw_response = response.text
                logger.info(f"Raw Assisto API response length: {len(raw_response)}")
                
                try:
                    response_data = response.json()
                    if isinstance(response_data, dict) and 'result' in response_data and response_data['result']:
                        speaker_texts = {}
                        for item in response_data['result']:
                            speaker = item.get('speaker', '0')
                            message = item.get('message', '').strip()
                            if message:
                                if speaker in speaker_texts:
                                    speaker_texts[speaker] += ' ' + message
                                else:
                                    speaker_texts[speaker] = message
                        combined_transcript = ' '.join(speaker_texts.values()) or json.dumps(response_data)
                        logger.info(f"Combined transcript from Assisto length: {len(combined_transcript)}")
                        return {"status": "success", "transcription": combined_transcript, "response": response_data}
                    else:
                        logger.warning(f"Unexpected response format from Assisto: {response_data}")
                        return {"status": "success", "transcription": raw_response, "response": response_data}
                except ValueError:
                    if len(response.text.strip()) > 50:
                        logger.warning(f"Non-JSON response from Assisto: {response.text[:200]}...")
                        return {"status": "success", "transcription": response.text, "response": response.text}
                    return {"status": "success", "transcription": None, "response": response.text}
            else:
                logger.error(f"Assisto API failed with status {response.status_code}: {response.text}")
                return {"status": "error", "response": f"API request failed with status code {response.status_code}"}
    except requests.RequestException as e:
        logger.error(f"Error during Assisto API request: {e}")
        return {"status": "error", "response": f"Error during API request: {e}"}
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return {"status": "error", "response": f"File not found: {filename}"}
    except Exception as e:
        logger.error(f"Unexpected error in process_audio_with_assisto: {e}")
        return {"status": "error", "response": f"Unexpected error: {e}"}

def process_audio_from_cos(file_key):
    errors = []
    session.clear()
    try:
        logger.info(f"Processing file from COS: {file_key}")
        json_data = get_json_from_cos(COS_BUCKET, file_key)
        if not json_data:
            errors.append("Failed to retrieve or parse JSON data.")
            return errors, None, None, None, None
        
        audio_url = json_data.get('AudioFile', '').strip()
        apikey = json_data.get('Apikey')
        
        if not audio_url or not urllib.parse.urlparse(audio_url).scheme:
            logger.error(f"Skipping {file_key} due to invalid AudioFile URL: {audio_url}")
            errors.append(f"Invalid or missing AudioFile URL in {file_key}")
            return errors, None, None, None, None
        
        # Step 1: Download audio (sequential, as it's a prerequisite)
        logger.info(f"Downloading audio from {audio_url}")
        filename = download_audio_file(audio_url)
        if not filename:
            errors.append("Failed to download audio file.")
            return errors, None, None, None, None
        
        # Step 2: Process audio with Assisto API (sequential, as it's a prerequisite)
        logger.info(f"Processing audio file {filename} with Assisto API")
        assisto_result = process_audio_with_assisto(filename, apikey, json_data)
        if assisto_result["status"] == "error":
            errors.append(f"Assisto API error: {assisto_result['response']}")
            return errors, None, None, None, None
        transcript = assisto_result.get("transcription", "No transcription available.")
        logger.info(f"Assisto transcription length: {len(transcript)}")
        
        # Clean up temporary file
        try:
            os.remove(filename)
            logger.info(f"Temporary file {filename} deleted.")
        except OSError as e:
            logger.error(f"Failed to delete temporary file: {e}")
        
        # Step 3: Run Separatespeakers, Getinsights, and Getcallquality in parallel
        def run_separatespeakers(trans):
            return Separatespeakers(trans)

        def run_getinsights(trans):
            return Getinsights(trans)

        def run_getcallquality(trans):
            return Getcallquality(trans)

        with ThreadPoolExecutor(max_workers=3) as executor:  # 3 workers for 3 tasks
            # Submit all tasks to run in parallel
            future_separated = executor.submit(run_separatespeakers, transcript)
            future_insights = executor.submit(run_getinsights, transcript)
            future_callquality = executor.submit(run_getcallquality, transcript)

            # Wait for all tasks to complete and collect results
            separated_transcript = future_separated.result()
            insights = future_insights.result()
            callquality = future_callquality.result()

        logger.info(f"Separated transcript length: {len(separated_transcript)}")
        logger.info(f"Insights: {insights}")
        logger.info(f"Call quality: {callquality}")
        
        # Step 4: Create JSON output
        create_json_output(file_key, transcript, insights, callquality, separated_transcript)
        return errors, separated_transcript, insights, callquality, separated_transcript
    
    except Exception as e:
        logger.error(f"Unexpected error in process_audio_from_cos: {e}")
        errors.append(f"Error processing {file_key}: {e}")
        return errors, None, None, None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    session.setdefault('transcript', "Select a JSON file to get call transcript")
    session.setdefault('insights', "Select a JSON file to get insights")
    session.setdefault('callquality', "Select a JSON file to get call quality")
    session.setdefault('separated', "Select a JSON file to get separated transcript")
    session.setdefault('json_filename', None)
    session.setdefault('json_path', None)

    try:
        json_files = get_cos_files()
    except Exception as e:
        logger.error(f"Error fetching COS files: {e}")
        json_files = []

    errors = []
    if request.method == 'POST':
        selected_file = request.form.get('selected_file')
        if selected_file and selected_file in json_files:
            logger.info(f"Processing selected file: {selected_file}")
            errors, separated, insights, callquality, _ = process_audio_from_cos(selected_file)
            if not errors and separated and separated != "No valid transcription available.":
                session['separated'] = separated
                if isinstance(insights, list) and len(insights) == 2:
                    session['insights'] = f"Insights Summary: {insights[0]}\nSentiment: {insights[1]}"
                else:
                    session['insights'] = "No insights available."
                session['callquality'] = callquality or "No call quality available."
                logger.info(f"Updated session - separated transcript length: {len(session['separated'])}")
            else:
                session['separated'] = f"Error: {' '.join(errors) if errors else 'No valid transcription available.'}"
                logger.error(f"Errors or no valid transcript: {errors or 'None'}")
            session['last_processed'] = selected_file
            return redirect(url_for('index'))
        else:
            errors.append("Invalid file selection or file not found.")

    if 'last_processed' in session and session['last_processed'] in json_files:
        selected_file = session['last_processed']
    else:
        selected_file = None

    return render_template('index.html', 
                         json_files=json_files, 
                         call_transcript=session['separated'],
                         call_insights=session['insights'],
                         call_quality=session['callquality'],
                         failed_files=[], 
                         errors=errors,
                         json_filename=session['json_filename'],
                         selected_file=selected_file)

@app.route('/api/process', methods=['POST'])
def api_process():
    try:
        data = request.get_json()
        if not data or 'file_key' not in data:
            return jsonify({"error": "Missing file_key in request"}), 400
        file_key = data['file_key']
        json_files = get_cos_files()
        if file_key not in json_files:
            return jsonify({"error": f"Invalid file_key: {file_key} not found in {len(json_files)} files"}), 400
        logger.info(f"Processing API request for file: {file_key}")
        errors, separated, insights, callquality, transcript = process_audio_from_cos(file_key)
        if errors:
            return jsonify({"error": errors[0], "status": "failed"}), 400
        insight_summary, sentiment = insights
        response = {
            "data": {
                "call_transcription": {
                    "raw_transcript": transcript,
                    "separated_transcript": separated
                },
                "call_insight": {
                    "summary": insight_summary
                },
                "call_rating": {
                    "quality": callquality
                },
                "call_sentiment_analysis": {
                    "sentiment": sentiment
                }
            }
        }
        logger.info(f"API response: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Exception in api_process: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/download_json')
def download_json():
    try:
        if 'json_path' in session and os.path.exists(session['json_path']):
            return send_file(session['json_path'], as_attachment=True, download_name=session['json_filename'])
        logger.warning("No JSON file available for download")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error downloading JSON: {e}")
        return redirect(url_for('index'))

if __name__ == '__main__':
    try:
        port = int(os.getenv('PORT', 8000))
        logger.info(f"Starting Flask app on port {port}")
        app.run(debug=False, host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start Flask app: {e}")
        raise