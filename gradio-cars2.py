import gradio as gr
from roboflow import Roboflow
import openai
import cv2
import os

# Ensure model initialization happens only once here
rf = Roboflow(api_key="9p4Y2dY8Y6KAT73koAbq")
project = rf.workspace().project("damaged-vehicle-images")
roboflow_model = project.version(3).model  # Renaming to avoid any potential overwrite

openai.api_key = "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"

def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        temperature=0,
        messages=[
            {"role": "system", "content": f"{prompt}"}
        ],
    )
    message = response.choices[0].message.content
    return message

def vehicle_assessment(make, model, year, image):
    # Save the image to a temporary file
    temp_filename = "temp_uploaded_image.jpg"
    cv2.imwrite(temp_filename, image)
    
    result = roboflow_model.predict(temp_filename, confidence=40, overlap=30).json()

    pred_l = []
    for prediction in result['predictions']:
        pred = prediction['class'].replace("_", " ")
        pred_l.append(pred)

    # Generate response using OpenAI
    prompt = f"""I have this data:
        Make	Model	Year	Type of Damage	Repair Description	Severity	Estimated Labor Hours	Parts Required	Estimated Parts Cost	Labor Rate	Total Estimated Cost
        Toyota	Camry	2015	Front Bumper	Dent Repair	Minor	2	Bumper, Paint	$200	$50/hr	$300 
        Honda	Civic	2010	Windshield	Replacement	Major	3	Windshield, Adhesive	$250	$45/hr	$385
        BMW	X5	2019	Rear Bumper	Scratch Repair	Minor	1.5	Bumper, Paint	$180	$65/hr	$277.5
        etc 

        and I have this info about a car:
        {make} , {model} , {year} ,{pred_l}
        predict the rest of the classes in a similar fashion to the data provided, Repair Description	Estimated Labor Hours	Parts Required	Estimated Parts Cost	Labor Rate	Total Estimated Cost

        DO NOT write sepertae estimaetuon for each damage, make your predictions based on all of them, so for example, if the damages ar like this:
        severe scratch, severe scratch, medium deformation, severe scratch, severe scratch, severe scratch ... etc,
        I want the output like this:
        Repair Description: Bodywork and Paint Repair
        Estimated Labor Hours: 4
        Parts Required: Paint, Body Filler, Replacement Parts
        Estimated Parts Cost: $400
        Labor Rate: $45/hr
        Total Estimated Cost: $800 
        
        this way you covered all damages and their estimation and you gave me the total thing which I wants
        
        """
    
    response = generate_response(prompt)
    
    # Clean up the temporary file
    os.remove(temp_filename)
    return response


with gr.Blocks() as app:
    gr.Image(type="pil", value='misbah-green-yellow-logo1.png', width=100, height=100)


    # gr.Markdown(
    #     '<div style="display: inline-block;width: 100%;">'
    #     '<img class="img-fluid AnimatedLogo" alt="misbah logo icons" title="misbah logo icons" src="https://drive.google.com/file/d/1hF9of-KUkaIknEE0DNgf3ScyMGbSUP5-/view?usp=drive_link" width="150px">'
    #     '</div>'
    # )

    # Define the layout using the Tabs
    with gr.Tab("Damage Assessment"):

        make_input = gr.inputs.Textbox(label="Make")
        model_input = gr.inputs.Textbox(label="Model")
        year_input = gr.inputs.Textbox(label="Year")
        image_input = gr.inputs.Image(label="Upload Image")
        vehicle_output = gr.outputs.Textbox(label="Prediction")
        vehicle_button = gr.Button("Assess")
    

    with gr.Tab("Repair Pal Chat"):
        chat_input = gr.inputs.Textbox(label="Chat Input")
        chat_output = gr.outputs.Textbox(label="Chat Output")
        chat_button = gr.Button("Send")

    with gr.Tab("Tab 3"):
        gr.Markdown("Content for Tab 3")

    def chat_with_bot(chat_input):
        chat_response = generate_response(chat_input)
        return chat_response

    chat_button.click(chat_with_bot, inputs=[chat_input], outputs=chat_output)

    # Link the button click to the vehicle assessment function
    vehicle_button.click(vehicle_assessment, inputs=[make_input, model_input, year_input, image_input], outputs=vehicle_output)

app.launch()
