# import python_tsp.heuristics
# print(dir(python_tsp.heuristics))

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch
import os
import re
import csv
from parser_for_coordinates import parse_points
from draw_circles import draw_dots_and_lines_on_image
from recalculate_to_latlon import recalculate_coordinates, percentage_to_lat_lon, read_coordinates_from_csv
from time import time
from python_tsp.heuristics import solve_tsp_local_search
import numpy as np

#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
mission_directory = 'created_missions'
if not os.path.exists(mission_directory):
    os.makedirs(mission_directory)

identified_new_data_directory = 'identified_new_data'

# Check if the directory exists, if not, create it
if not os.path.exists(identified_new_data_directory):
    os.makedirs(identified_new_data_directory)


list_of_the_resulted_coordinates_percentage = []
list_of_the_resulted_coordinates_lat_lon = []
list_of_optimized_coordinates = []

NUMBER_OF_SAMPLES = 30 #len(os.listdir('/VLM_Drone/dataset_images'))
print('NUMBER_OF_SAMPLES',NUMBER_OF_SAMPLES)

# load the processor
processor = AutoProcessor.from_pretrained(
    'cyan2k/molmo-7B-O-bnb-4bit',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'cyan2k/molmo-7B-O-bnb-4bit',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

llm = ChatOpenAI(api_key='', model_name='gpt-4o', temperature=0)

# 1. Step 1: Extract object types from the user's input command using the LLM

step_1_template = """
Extract all types of objects the drone needs to find from the following mission description:
"{command}"

Output the result in JSON format with a list of object types.
Example output:
{{
    "object_types": ["village", "airfield", "stadium", "tennis court", "building", "ponds", "crossroad", "roundabout"]
}}
"""

step_1_prompt = PromptTemplate(input_variables=["command"], template=step_1_template)

step_1_chain = step_1_prompt | llm

example_objects = '''
{
        "village_1": {"type": "village", "coordinates": [1.5, 3.5]},
        "village_2": {"type": "village", "coordinates": [2.5, 6.0]},
        "airfield": {"type": "airfield", "coordinates": [8.0, 6.5]}
    }
'''

2. Step 2: Use Molmo model to find objects on the map
def find_objects(json_input, example_objects):
    """
    Process the mission description to find object coordinates on the map and return optimized coordinates using TSP.
    """
    search_string = str()
    find_objects_json_input = json_input.replace("`", "").replace("json","")    #[9::-3]
    
    find_objects_json_input_2 = json.loads(find_objects_json_input)

    for i in range(0, len(find_objects_json_input_2["object_types"])):
        sample = find_objects_json_input_2["object_types"][i]
        search_string = search_string + sample

    for i in range(1, NUMBER_OF_SAMPLES+1):
        print(f"Processing image {i}")
        string = f'benchmark-UAV-VLPA-nano-30/images/{i}.jpg' 
        
        # Process the image and text
        inputs = processor.process(
            images=[Image.open(f'benchmark-UAV-VLPA-nano-30/images/{i}.jpg')],
            text=f'''
            This is the satellite image of a city. Please, point all the next objects: {sample} 
            '''
        )

        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # Generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        parsed_points = parse_points(generated_text)

        image_number = i

        csv_file_path = 'benchmark-UAV-VLPA-nano-30/parsed_coordinates.csv'
        coordinates_dict = read_coordinates_from_csv(csv_file_path)

        result_coordinates = recalculate_coordinates(parsed_points, image_number, coordinates_dict)
        draw_dots_and_lines_on_image(f'benchmark-UAV-VLPA-nano-30/images/{i}.jpg', parsed_points, output_path=f'identified_new_data/identified{i}.jpg')

        print(result_coordinates)

        list_of_the_resulted_coordinates_percentage.append(parsed_points)
        list_of_the_resulted_coordinates_lat_lon.append(result_coordinates)

        # Optimize coordinates using TSP and append to the list
        optimized_coordinates = tsp_optimized_coordinates(result_coordinates)
        list_of_optimized_coordinates.append(optimized_coordinates)

    return json.dumps(result_coordinates), list_of_the_resulted_coordinates_percentage, list_of_the_resulted_coordinates_lat_lon, list_of_optimized_coordinates


# 3. Step 3: Generate flight plan using LLM and identified objects
step_3_template = """
Given the mission description: "{command}" and the following identified objects: {objects}, generate a flight plan in pseudo-language.

The available commands are on the website:


Some hints:
- arm throttle: arm the copter
- takeoff Z: lift Z meters
- disarm: disarm the copter
- mode rtl: return to home
- mode circle: circle and observe at the current position
- mode guided(X Y Z): fly to the specified location

Use the identified objects to create the mission.

Provide me only with commands string-by-string.
"""

step_3_prompt = PromptTemplate(input_variables=["command", "objects"], template=step_3_template)
step_3_chain = step_3_prompt | llm

# TSP Function: Solve TSP using Euclidean distance between coordinates
def tsp_optimized_coordinates(coordinates):
    """
    Use TSP to optimize the path for a drone using Euclidean distance between coordinates.
    The coordinates should be in a dictionary format where each key represents a building 
    and each value is another dictionary with 'type' and 'coordinates' as keys.
    """
    coords = []
    names = []
    
    # Extract coordinates and names from the input dictionary
    for name, data in coordinates.items():
        if 'coordinates' in data:
            coords.append(data['coordinates'])
            names.append(name)
    
    # Calculate Euclidean distance matrix
    num_coords = len(coords)
    distance_matrix = np.zeros((num_coords, num_coords))

    for i in range(num_coords):
        for j in range(i + 1, num_coords):
            dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            distance_matrix[i][j] = distance_matrix[j][i] = dist

    # Solve the TSP using local search
    optimal_order = solve_tsp_local_search(distance_matrix)

    # Ensure that the optimal_order is a list of indices, not a list of lists
    if isinstance(optimal_order[0], list):  # This checks if it's a nested list
        optimal_order = optimal_order[0]  # Flatten the list

    # Reorder the coordinates according to the optimal TSP path
    optimized_coordinates = {names[i]: coordinates[names[i]] for i in optimal_order}

    return optimized_coordinates



Full pipeline function
def generate_drone_mission(command):
    # Step 1: Extract object types
    object_types_response = step_1_chain.invoke({"command": command})
    object_types_json = object_types_response.content  # Use 'content' to get the actual response text

    # Step 2: Find objects on the map
    t1_find_objects = time()
    objects_json, list_of_the_resulted_coordinates_percentage, list_of_the_resulted_coordinates_lat_lon, list_of_optimized_coordinates = find_objects(object_types_json, example_objects)
    t2_find_objects = time()

    del_t_find_objects = (t2_find_objects - t1_find_objects) / 60
    print('Length of optimized coordinates:', len(list_of_optimized_coordinates))

    #print('objects_json =', objects_json)

    
    # Step 3: Generate the flight plan
    t1_generate_drone_mission = time()

    for i in range(1,len(list_of_optimized_coordinates)+1): 
        flight_plan_response = step_3_chain.invoke({"command": command, "objects": list_of_optimized_coordinates[i-1]})
    #print('flight_plan_response = ', flight_plan_response)
        with open(f"created_missions/mission{i}.txt","w") as file:   
            file.write(str(flight_plan_response.content))

        print(flight_plan_response.content)

    t2_generate_drone_mission = time()
    del_t_generate_drone_mission = (t2_generate_drone_mission - t1_generate_drone_mission)/60


    return flight_plan_response.content, del_t_find_objects, del_t_generate_drone_mission  # Return the response text from AIMessage


# Example usage:
command = """Create a flight plan for the quadcopter to fly around each of the building at the height 100m return to home and land at the take-off point."""


# Run the full pipeline
flight_plan, vlm_model_time, mission_generation_time = generate_drone_mission(command)
total_computational_time = vlm_model_time + mission_generation_time

# Evaluation time
print('-------------------------------------------------------------------')
print('Time to get VLM results: ', vlm_model_time, 'mins')
print('Time to get Mission Text files: ', mission_generation_time, 'mins')
print('Total Computational Time: ', total_computational_time, 'mins')
