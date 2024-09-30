# Import dependencies    
import datetime    
import json    
import time    
import os    
import shutil
from openai import AzureOpenAI    
from dotenv import load_dotenv    
import copy    
import textwrap    
import requests    
import threading    
import streamlit as st    
from queue import Queue  
from PIL import Image
import base64
import io
import fitz  # PyMuPDF
from process_inputs import process_inputs  
# Load environment variables    
load_dotenv("./.env")    

# Define a flag to toggle deletion of the temporary folder
DELETE_TEMP_FOLDER = os.getenv("DELETE_TEMP_FOLDER", "true").lower() == "true"
TEMP_FOLDER = "./temp_uploads"

# Function to read XML file content as a string  
def load_use_case_from_file(file_path):  
    with open(file_path, 'r') as file:  
        return file.read()  
    
# Define function for calling GPT4o with streaming  
def gpt4o_call(system_message, prompt, result_dict, queue):    
    client = AzureOpenAI(    
        api_version=os.getenv("4oAPI_VERSION"),    
        azure_endpoint=os.getenv("4oAZURE_ENDPOINT"),    
        api_key=os.getenv("4oAPI_KEY")    
    )    
    
    start_time = time.time()    
    
    completion = client.chat.completions.create(    
        model=os.getenv("4oMODEL"),    
        messages=[    
            {"role": "system", "content": system_message},    
            {"role": "user", "content": prompt},    
        ],    
        stream=True  # Enable streaming  
    )    
    
    response_text = ""  
    for chunk in completion:  
        if chunk.choices and chunk.choices[0].delta.content:  
            response_text += chunk.choices[0].delta.content  
            queue.put(response_text)  
    
    elapsed_time = time.time() - start_time    
    
    result_dict['4o'] = {    
        'response': response_text,    
        'time': elapsed_time    
    }    
    queue.put(f"Elapsed time: {elapsed_time:.2f} seconds")    
    
# Define function for calling O1 API  
def call_o1_api(system_message, user_message):  
    prompt = system_message + user_message    
    
    url = os.getenv("o1AZURE_ENDPOINT")    
    headers = {    
        "api-key": os.getenv("o1API_KEY"),    
        "Content-Type": "application/json"    
    }    
    data = {    
        "messages": [    
            {    
                "role": "user",    
                "content": [    
                    {"type": "text", "text": prompt}    
                ],    
            }    
        ]    
    }    
    
    start_time = time.time()    
    
    response = requests.post(url, headers=headers, json=data)    
    response_json = response.json()    
    messageo1 = response_json["choices"][0]["message"]["content"]    
    
    elapsed_time = time.time() - start_time    
    
    return messageo1, elapsed_time  
  
# Define function for calling O1 and storing the result  
def o1_call(system_message, user_message, result_dict):    
    response, elapsed_time = call_o1_api(system_message, user_message)  
    result_dict['o1'] = {    
        'response': response,    
        'time': elapsed_time    
    }    
  
# Define function for comparing responses using O1  
def compare_responses(response_4o, response_o1):  
    system_message = "You are an expert reviewer, who is helping review two candidates responses to a question."  
    user_message = f"Compare the following two responses and summarize the key differences:\n\nResponse 1 GPT-4o Model:\n{response_4o}\n\nResponse 2 o1 Model:\n{response_o1}. Generate a succinct comparison, and call out the key elements that make one response better than another. Be critical in your analysis."  
    comparison_result, _ = call_o1_api(system_message, user_message)  
      
    return comparison_result  

# Function to process images and convert them to text using GPT-4o
def process_images(images):
    client = AzureOpenAI(
        api_version=os.getenv("4oAPI_VERSION"),
        azure_endpoint=os.getenv("4oAZURE_ENDPOINT"),
        api_key=os.getenv("4oAPI_KEY")
    )

    descriptions = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        system_prompt = "Generate a highly detailed text description of this image, making sure to capture all the information within the image as words. If there is text, tables or other text based information, include this in a section of your response as markdown."
        response = client.chat.completions.create(
            model=os.getenv("4oMODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Here is the input image:"},
                    {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{img_str}', "detail": "low"}}
                ]}
            ],
            temperature=0,
        )
        descriptions.append(response.choices[0].message.content)
    
    return descriptions

def process_pdf(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num + 1}.jepg")
        img.save(image_path, "JPEG")

# Streamlit app    
def main():    
    st.set_page_config(page_title="o1 vs 4o Use Case Comparison", layout="wide")    
    st.title("o1 vs 4o Use Case Comparison")    
    
    # Load use case from file  
    use_case_path = './use-cases/home-insurance-claim/Claim Processing.xml'  
    home_insurance_claim = load_use_case_from_file(use_case_path)  
    
    # Use case dropdown    
    use_cases = {    
        "Greeting": "Hello, how are you?",    
        "Weather Inquiry": "What's the weather like today?",    
        "Joke Request": "Tell me a joke.",    
        "Home Insurance Claim": home_insurance_claim  
        # Add more prebuilt use cases as needed    
    }    
    
    selected_use_case = st.selectbox("Select a use case:", list(use_cases.keys()))    
    default_input = use_cases[selected_use_case]    
    
    # Input box (takes up the width of the screen)    
    user_input = st.text_area("Enter your input:", value=default_input, height=100)    

    # Section to upload supporting documents
    st.subheader("Upload Supporting Documents")
    uploaded_files = st.file_uploader("Choose images or PDFs", accept_multiple_files=True, type=["jpg", "jpeg", "png", "pdf"])
  
    # Process uploaded files  
    if uploaded_files and st.button("Upload Files"):  
        # Process the files and generate descriptions  
        process_inputs(uploaded_files)  
  
        # Load images and descriptions from TEMP_FOLDER  
        image_files = [os.path.join(TEMP_FOLDER, f) for f in os.listdir(TEMP_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]  
        descriptions = []  
        for img_file in image_files:  
            base_name = os.path.splitext(os.path.basename(img_file))[0]  
            description_path = os.path.join(TEMP_FOLDER, f"{base_name}.txt")  
            with open(description_path, 'r', encoding='utf-8') as f:  
                description = f.read()  
                descriptions.append((img_file, description))  
  
        if descriptions:  
            st.session_state['combined_text'] = user_input + "\n\n" + "\n\n".join([desc for _, desc in descriptions])  
            st.session_state['descriptions'] = descriptions
        else:  
            st.session_state['combined_text'] = user_input  
            st.session_state['descriptions'] = []
  
    # Display images as tiles with descriptions  
    if 'descriptions' in st.session_state:
        st.subheader("Uploaded Images and Descriptions")  
        cols = st.columns(3)  
        fixed_height = 300  # Set a fixed height for the combined image and text  
  
        for i, (img_path, description) in enumerate(st.session_state['descriptions']):  
            image = Image.open(img_path)  
            col = cols[i % 3]  
            with col.container():  
                buffered = io.BytesIO()  
                image.save(buffered, format="JPEG")  
                img_str = base64.b64encode(buffered.getvalue()).decode()  
                col.markdown(  
                    f"""  
                    <style>  
                        .image-tile {{  
                            height: {fixed_height}px;  
                            display: flex;  
                            flex-direction: column;  
                            justify-content: space-between;  
                            margin: 10px;  
                            border: 1px solid #ddd;  
                            border-radius: 8px;  
                            overflow: hidden;  
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  
                            transition: transform 0.2s;  
                        }}  
                        .image-tile:hover {{  
                            transform: scale(1.05);  
                        }}  
                        .image-container {{  
                            flex: 1;  
                            display: flex;  
                            align-items: center;  
                            justify-content: center;  
                            overflow: hidden;  
                        }}  
                        .image-container img {{  
                            max-height: {fixed_height - 100}px;  
                            width: auto;  
                        }}  
                        .description-container {{  
                            height: 100px;  
                            width: 100%;  
                            padding: 10px;  
                            background: #f9f9f9;  
                            border-top: 1px solid #ddd;  
                        }}  
                    </style>  
                    <div class='image-tile'>  
                        <div class='image-container'>  
                            <img src='data:image/jpeg;base64,{img_str}'>  
                        </div>  
                        <div class='description-container'>  
                            <textarea style='height: 100%; width: 100%; border: none; resize: none;' disabled>{description}</textarea>  
                        </div>  
                    </div>  
                    """,  
                    unsafe_allow_html=True  
                )  
  
    else:  
        # No uploaded files; use the user_input as combined_text  
        st.session_state['combined_text'] = user_input  
    
    # Button to submit    
    if st.button("Submit"):    
        # Display placeholders for responses    
        col1, col2 = st.columns(2)    
    
        with col1:    
            st.subheader("4o Model Response")    
            response_placeholder_4o = st.empty()    
            time_placeholder_4o = st.empty()    
    
        with col2:    
            st.subheader("o1 Model Response")    
            response_placeholder_o1 = st.empty()    
            time_placeholder_o1 = st.empty()    
    
        # Dictionary to store results    
        result_dict = {}    
        queue = Queue()  
    
        # Start threads for both API calls    
        threads = []    
        t1 = threading.Thread(target=gpt4o_call, args=("You are a helpful AI assistant.", st.session_state['combined_text'], result_dict, queue))    
        t2 = threading.Thread(target=o1_call, args=("You are a helpful AI assistant.", st.session_state['combined_text'], result_dict))    
        threads.append(t1)    
        threads.append(t2)    
        t1.start()    
        t2.start()    
    
        # Update the Streamlit UI with the streamed response  
        while t1.is_alive():  
            while not queue.empty():  
                response_placeholder_4o.write(queue.get())  
            time.sleep(0.1)  
    
        # Wait for both threads to complete    
        for t in threads:    
            t.join()    
    
        # Display the 4o response and elapsed time  
        with col1:  
            response_placeholder_4o.write(result_dict['4o']['response'])    
            time_placeholder_4o.write(f"Elapsed time: {result_dict['4o']['time']:.2f} seconds")  
    
        # Display the O1 response and elapsed time    
        with col2:    
            response_placeholder_o1.write(result_dict['o1']['response'])    
            time_placeholder_o1.write(f"Elapsed time: {result_dict['o1']['time']:.2f} seconds")    
    
        # Compare the responses and display the comparison  
        st.subheader("Comparison of Responses")  
        comparison_result = compare_responses(result_dict['4o']['response'], result_dict['o1']['response'])  
        st.write(comparison_result)  
    


if __name__ == "__main__":
    # Clean up the temporary folder if the flag is set
    if DELETE_TEMP_FOLDER and os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)    
    main()