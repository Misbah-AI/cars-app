import gradio as gr
from roboflow import Roboflow
import openai
import cv2
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Retrieve API keys from environment variables
# ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if the environment variables are loaded properly
if not OPENAI_API_KEY:
    raise ValueError("API keys not found in .env file")
# Ensure model initialization happens only once here
rf = Roboflow(api_key="9p4Y2dY8Y6KAT73koAbq")
project = rf.workspace().project("damaged-vehicle-images")
roboflow_model = project.version(3).model  # Renaming to avoid any potential overwrite

openai.api_key = OPENAI_API_KEY

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
      
        follow this structure:

        Car Damage Assessment and Estimation Report

Vehicle Information:

    Make: [Vehicle Make]
    Model: [Vehicle Model]
    Year: [Vehicle Year]

Damage Description:

    Description of Damage: [Describe the damage in detail]

Estimated Repair Costs:
    
    Repair Description: Bodywork and Paint Repair ... etc
    Estimated Labor Hours: [Labor hours Estimate]
    Parts Required: Paint, Body Filler, Replacement Parts ... etc
    Estimated Parts Cost: $[Parts Cost Estimate]
    Labor Rate: $[Labor Cost Estimate]/hr
    Paint and Materials: $[Paint and Materials Cost Estimate]
    Total Estimated Repair Cost: $[Total Estimated Cost]
        

this way you covered all damages and their estimation and you gave me the total thing which I wants
        
        """
    
    response = generate_response(prompt)
    
    # Clean up the temporary file
    os.remove(temp_filename)
    return response

with gr.Blocks() as app:
    # gr.Image(type="pil", value='misbah-green-yellow-logo1.png', width=100, height=100)


    gr.Markdown(
        '<div style="display: inline-block;width: 100%;">'
        '<img class="img-fluid AnimatedLogo" alt="misbah logo icons" title="misbah logo icons" src="https://img1.wsimg.com/isteam/ip/8a73a484-6b7c-4a18-9dc5-2526353c4068/misbah-green-yellow-logo1.png/:/rs=w:275,h:200,cg:true,m/cr=w:275,h:200/qt=q:95" width="150px">'
        '</div>'
    )

    # Define the layout using the Tabs
    with gr.Tab("Damage Assessment"):
        gr.Markdown('## Evaluate vehicle damage and estimate repair costs by inputting vehicle details and uploading an image of the damage. Receive a comprehensive repair estimate.')
        make_input = gr.inputs.Textbox(label="Make")
        model_input = gr.inputs.Textbox(label="Model")
        year_input = gr.inputs.Textbox(label="Year")
        image_input = gr.inputs.Image(label="Upload Image")
        vehicle_output = gr.outputs.Textbox(label="Estimation Report")
        vehicle_button = gr.Button("Assess")
    

    with gr.Tab("Repair Pal Chat"):
        gr.Markdown('## chat with our Repair Pal about your car problem')

        system_message = {"role": "system", "content": "You are a helpful assistant."}
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label='write your problem')
        clear = gr.Button("Clear")

        state = gr.State([])

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history, messages_history):
            user_message = history[-1][0]
            bot_message, messages_history = ask_gpt(user_message, messages_history)
            messages_history += [{"role": "assistant", "content": bot_message}]
            history[-1][1] = bot_message
            time.sleep(1)
            return history, messages_history

        def ask_gpt(message, messages_history):
            messages_history += [{"role": "user", "content": message}]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=messages_history
            )
            return response['choices'][0]['message']['content'], messages_history

        def init_history(messages_history):
            messages_history = []
            messages_history += [system_message]
            return messages_history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, [chatbot, state], [chatbot, state]
        )

        clear.click(lambda: None, None, chatbot, queue=False).success(init_history, [state], [state])

    def chat_with_bot(chat_input):
        chat_response = generate_response(chat_input)
        return chat_response

    

 
    with gr.Tab("Repair shops and spare parts"):
        gr.Markdown(""" 
                    ## Based on the damage assesment report, you can repair your car at these repair shops:

                    [Location](https://maps.app.goo.gl/DwZRi2Kr18fYBQR96?g_st=ic)

                    [Location](https://maps.app.goo.gl/knkiyaLMqYLM86H67?g_st=ic)

                    [Location](https://maps.app.goo.gl/xQSQMytHA3QmshMn6?g_st=ic)

                   ## You can find spare parts for your damaged parts at these websites: 
                    
                    [Rafraf](https://rafraf.com/)
                    
                    [Afyal](https://afyal.com/) 
                    
                    """)



    # Link the button click to the vehicle assessment function
    vehicle_button.click(vehicle_assessment, inputs=[make_input, model_input, year_input, image_input], outputs=vehicle_output)
    # chat_button.click(chat_with_bot, inputs=[chat_input], outputs=chat_output)

    
app.launch(share=True)
