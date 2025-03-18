# UAV-VLPA*: A Vision-Language-Path-Action System for Optimal Route Generation on a Large Scales


---

## Table of Contents
1. [Abstract](#abstract)
2. [Benchmark](#benchmark)
3. [Installation](#installation)
4. [Path-Plans Creation](#path-plan-creation)
5. [Experimental Results](#experimental-results)
6. [Simulation Video](simulation-video)
7. [Citation](citation)


---
## Abstract
The UAV-VLPA* (Visual-Language-Planning-and-Action) system represents a cutting-edge advancement in aerial robotics, designed to enhance communication and operational efficiency for unmanned aerial vehicles (UAVs). By integrating advanced planning capabilities, the system addresses the Traveling Salesman Problem (TSP) to optimize flight paths, reducing the total trajectory length by 18.5\% compared to traditional methods. Additionally, the incorporation of the A* algorithm enables robust obstacle avoidance, ensuring safe and efficient navigation in complex environments. The system leverages satellite imagery processing combined with the Visual Language Model (VLM) and GPT's natural language processing capabilities, allowing users to generate detailed flight plans through simple text commands. This seamless fusion of visual and linguistic analysis empowers precise decision-making and mission planning, making UAV-VLPA* a transformative tool for modern aerial operations. With its unmatched operational efficiency, navigational safety, and user-friendly functionality, UAV-VLPA* sets a new standard in autonomous aerial robotics, paving the way for future innovations in the field.



This repository includes:
- The implementation of the UAV-VLPA* framework.


### UAV-VLPA* Framework


<div align="center">
  <img src="https://github.com/user-attachments/assets/45639ac5-b05e-43e3-badc-bed35e011a51" alt="system_architecture" width="600"/>
</div>





## Benchmark

The images of the benchmark are stored in the folder ```benchmark-UAV-VLPA-nano-30/images```. The metadata files are ```benchmark-UAV-VLPA-nano-30/img_lat_long_data.txt``` and ```benchmark-UAV-VLPA-nano-30/parsed_coordinates.csv```.

## Installation



To install requirements, run 

```
pip -r requirements.txt
```
!12GB VRAM minimum

## Export your ChatGpt api key
```
export api_key="your chatgpt ap_key"
```

## Mission generation

To generate commands for UAV add your API key for ChatGPT in the generate_plans.py, then run
```
python3 AStar.py
```
or
```
python3 tsp.py
```
```
python3 run_vlm.py
```

## Path-Plans Creation


Some examples of the path generated can be seen below:

<div align="center">
  <img src="https://github.com/user-attachments/assets/97beebc4-f579-46f8-9544-64f1b1002675" alt="examples_path_generated" width="900"/>
</div>



### Trajectory Bar Chart of UAV-VLA:

<div align="center">
  <img src="https://github.com/user-attachments/assets/e27a0c86-e54a-433a-822c-dc68297fdd37" alt="traj_bar_chart" width="600"/>
</div>


### Trajectory Bar Chart of UAV-VLPA*:

<div align="center">
  <img src="https://github.com/user-attachments/assets/11cc66e9-ec0f-4bd4-baad-a3c24fd4f52a" alt="traj_bar_chart" width="600"/>
</div>

### Error Box Plot:

<div align="center">
  <img src="https://github.com/user-attachments/assets/002077a0-5582-4643-b6b4-84f07b1eec77" alt="error_box_plot" width="500"/>
</div>

