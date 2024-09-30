# Import dependencies  
import os  
import io  
import base64  
import fitz  # PyMuPDF  
from PIL import Image  
from openai import AzureOpenAI  
from dotenv import load_dotenv  
  
# Load environment variables  
load_dotenv("./.env")  
  
# Constants  
TEMP_FOLDER = "./temp_uploads"  
  
# Function to process PDFs and convert them to images  
def process_pdf(pdf_path, output_folder):  
    pdf_document = fitz.open(pdf_path)  
    for page_num in range(len(pdf_document)):  
        page = pdf_document.load_page(page_num)  
        pix = page.get_pixmap()  
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  
        image_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num + 1}.jpeg")  
        img.save(image_path, "JPEG")  
  
# Function to process images and generate descriptions  
def process_images_and_generate_descriptions(image_files, output_folder):  
    client = AzureOpenAI(  
        api_version=os.getenv("4oAPI_VERSION"),  
        azure_endpoint=os.getenv("4oAZURE_ENDPOINT"),  
        api_key=os.getenv("4oAPI_KEY")  
    )  
    for image_file in image_files:  
        with Image.open(image_file) as image:  
            # Convert image to base64  
            buffered = io.BytesIO()  
            image.save(buffered, format="JPEG")  
            img_str = base64.b64encode(buffered.getvalue()).decode()  
  
            system_prompt = "Generate a highly detailed text description of this image, making sure to capture all the information within the image as words. If there is text, tables, or other text-based information, include this in a section of your response as markdown."  
  
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
  
            description = response.choices[0].message.content  
  
            # Save the description to a text file with the same base name as the image  
            base_name = os.path.splitext(os.path.basename(image_file))[0]  
            description_path = os.path.join(output_folder, f"{base_name}.txt")  
            with open(description_path, 'w', encoding='utf-8') as f:  
                f.write(description)  
  
            # Save the processed image to the output folder (if not already there)  
            if image_file != os.path.join(output_folder, os.path.basename(image_file)):  
                image.save(os.path.join(output_folder, os.path.basename(image_file)), "JPEG")  
  
# Main function to process uploaded files  
def process_inputs(uploaded_files):  
    if not os.path.exists(TEMP_FOLDER):  
        os.makedirs(TEMP_FOLDER)  
  
    for uploaded_file in uploaded_files:  
        file_path = os.path.join(TEMP_FOLDER, uploaded_file.name)  
        with open(file_path, "wb") as f:  
            f.write(uploaded_file.getbuffer())  
  
        if uploaded_file.type == "application/pdf":  
            process_pdf(file_path, TEMP_FOLDER)  
  
    # Gather all image files (including those generated from PDFs)  
    image_files = [os.path.join(TEMP_FOLDER, f) for f in os.listdir(TEMP_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]  
  
    # Process images and generate descriptions, saved to disk  
    process_images_and_generate_descriptions(image_files, TEMP_FOLDER)  