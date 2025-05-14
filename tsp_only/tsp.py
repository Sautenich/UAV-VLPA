"""
TSP (Traveling Salesman Problem) Solver for UAV Path Planning

This module implements a TSP solver using local search optimization to find
the optimal route for a UAV visiting multiple waypoints. It integrates with
vision and language models for waypoint identification and mission planning.
"""

from typing import Dict, List, Tuple, Any
import os
import json
import numpy as np
from PIL import Image
import torch
from python_tsp.heuristics import solve_tsp_local_search
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from parser_for_coordinates import parse_points
from draw_circles import draw_dots_and_lines_on_image
from recalculate_to_latlon import (
    recalculate_coordinates,
    percentage_to_lat_lon,
    read_coordinates_from_csv
)

# Configuration constants
NUMBER_OF_SAMPLES = 30
MISSION_DIR = 'created_missions'
IDENTIFIED_DATA_DIR = 'identified_new_data'
BENCHMARK_DIR = 'benchmark-UAV-VLPA-nano-30'

# Ensure required directories exist
os.makedirs(MISSION_DIR, exist_ok=True)
os.makedirs(IDENTIFIED_DATA_DIR, exist_ok=True)

class VisionLanguageProcessor:
    """Handles vision and language processing for waypoint identification."""
    
    def __init__(self):
        """Initialize vision and language models."""
        self.processor = AutoProcessor.from_pretrained(
            'cyan2k/molmo-7B-O-bnb-4bit',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            'cyan2k/molmo-7B-O-bnb-4bit',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        
        self.llm = ChatOpenAI(
            api_key=os.getenv('OPENAI_API_KEY', ''),
            model_name='gpt-4',
            temperature=0
        )

    def extract_object_types(self, command: str) -> Dict[str, List[str]]:
        """
        Extract object types from mission command using LLM.
        
        Args:
            command: Natural language mission description
            
        Returns:
            Dictionary containing list of object types to identify
        """
        template = """
        Extract all types of objects the drone needs to find from the following mission description:
        "{command}"

        Output the result in JSON format with a list of object types.
        Example output:
        {
            "object_types": ["village", "airfield", "stadium", "tennis court", "building", "ponds", "crossroad", "roundabout"]
        }
        """
        
        prompt = PromptTemplate(
            input_variables=["command"],
            template=template
        )
        
        chain = prompt | self.llm
        return json.loads(str(chain.invoke({"command": command})))

    def process_image(self, image_path: str, search_query: str) -> List[Tuple[float, float]]:
        """
        Process an image to identify objects and their coordinates.
        
        Args:
            image_path: Path to the image file
            search_query: Description of objects to identify
            
        Returns:
            List of (x, y) coordinates for identified objects
        """
        inputs = self.processor.process(
            images=[Image.open(image_path)],
            text=f'This is the satellite image of a city. Please, point all the next objects: {search_query}'
        )
        
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )
        
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        return parse_points(generated_text)

class TSPSolver:
    """Solves the Traveling Salesman Problem for UAV route optimization."""
    
    @staticmethod
    def optimize_route(coordinates: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Optimize the route using TSP solver.
        
        Args:
            coordinates: Dictionary of waypoints with their coordinates
                Format: {
                    'point_name': {
                        'type': str,
                        'coordinates': [float, float]
                    }
                }
                
        Returns:
            Dictionary of waypoints in optimized order
        """
        # Extract coordinates and names
        coords = []
        names = []
        for name, data in coordinates.items():
            if 'coordinates' in data:
                coords.append(data['coordinates'])
                names.append(name)
        
        # Calculate distance matrix
        num_coords = len(coords)
        distance_matrix = np.zeros((num_coords, num_coords))
        
        for i in range(num_coords):
            for j in range(i + 1, num_coords):
                dist = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
                distance_matrix[i][j] = distance_matrix[j][i] = dist
        
        # Solve TSP
        optimal_order = solve_tsp_local_search(distance_matrix)
        if isinstance(optimal_order[0], list):
            optimal_order = optimal_order[0]
        
        # Reorder coordinates
        return {
            names[i]: coordinates[names[i]]
            for i in optimal_order
        }

def generate_drone_mission(command: str) -> Tuple[List[Dict], List[List], List[Dict]]:
    """
    Generate a complete drone mission from a natural language command.
    
    Args:
        command: Natural language description of the mission
        
    Returns:
        Tuple containing:
        - List of percentage-based coordinates
        - List of latitude/longitude coordinates
        - List of optimized coordinates
    """
    processor = VisionLanguageProcessor()
    solver = TSPSolver()
    
    # Extract object types from command
    object_types = processor.extract_object_types(command)
    search_query = ' '.join(object_types['object_types'])
    
    # Process each image in the dataset
    percentage_coords = []
    latlon_coords = []
    optimized_coords = []
    
    coordinates_dict = read_coordinates_from_csv(
        f'{BENCHMARK_DIR}/parsed_coordinates.csv'
    )
    
    for i in range(1, NUMBER_OF_SAMPLES + 1):
        print(f"Processing image {i}")
        image_path = f'{BENCHMARK_DIR}/images/{i}.jpg'
        
        # Process image and get coordinates
        parsed_points = processor.process_image(image_path, search_query)
        result_coords = recalculate_coordinates(parsed_points, i, coordinates_dict)
        
        # Visualize results
        draw_dots_and_lines_on_image(
            image_path,
            parsed_points,
            output_path=f'{IDENTIFIED_DATA_DIR}/identified{i}.jpg'
        )
        
        # Store results
        percentage_coords.append(parsed_points)
        latlon_coords.append(result_coords)
        
        # Optimize route
        optimized = solver.optimize_route(result_coords)
        optimized_coords.append(optimized)
    
    return percentage_coords, latlon_coords, optimized_coords

if __name__ == "__main__":
    # Example usage
    mission_command = "Find all villages, airfields, and stadiums in the area"
    percentage_coords, latlon_coords, optimized_coords = generate_drone_mission(mission_command)
