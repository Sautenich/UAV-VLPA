# UAV-VLPA*: Vision-Language-Path-Action System for Optimal UAV Route Generation

A sophisticated system that combines computer vision, natural language processing, and path planning to generate optimized flight paths for UAVs (Unmanned Aerial Vehicles).

## Features

- **Visual Processing**: Analyzes satellite imagery to identify objects and landmarks
- **Natural Language Interface**: Converts text commands into actionable flight plans
- **Path Optimization**: 
  - TSP (Traveling Salesman Problem) solver for optimal waypoint ordering
  - A* pathfinding for obstacle avoidance
  - 18.5% reduction in total trajectory length compared to traditional methods
- **Multi-Modal Integration**: Seamlessly combines visual and linguistic analysis

## System Architecture

<div align="center">
  <img src="https://github.com/user-attachments/assets/45639ac5-b05e-43e3-badc-bed35e011a51" alt="system_architecture" width="600"/>
</div>

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU with 12GB+ VRAM
- CUDA Toolkit 11.0+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/UAV-VLPA.git
cd UAV-VLPA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your_chatgpt_api_key"
```

## Project Structure

```
UAV-VLPA/
├── pathplan/           # Path planning algorithms
│   ├── Astar.py       # A* implementation for obstacle avoidance
│   └── utils.py       # Utility functions for path planning
├── tsp_only/          # TSP optimization
│   └── tsp.py         # TSP solver implementation
├── benchmark-UAV-VLPA-nano-30/  # Benchmark dataset
│   ├── images/        # Satellite imagery
│   └── parsed_coordinates.csv   # Ground truth coordinates
└── requirements.txt    # Project dependencies
```

## Usage

1. **Generate Mission Plan**:
```bash
python3 generate_plans.py --mission "your mission description"
```

2. **Run Path Planning**:
```bash
python3 pathplan/Astar.py
```

3. **Run TSP Optimization**:
```bash
python3 tsp_only/tsp.py
```

4. **Process Visual Data**:
```bash
python3 run_vlm.py
```

## Example Results

### Path Generation Examples

<div align="center">
  <img src="https://github.com/user-attachments/assets/84fb1e8e-4926-4e73-bf7f-fbb83a3fdc33" alt="examples_path_generated2" width="600"/>
</div>

### Performance Comparison

#### UAV-VLA Trajectory
<div align="center">
  <img src="https://github.com/user-attachments/assets/e27a0c86-e54a-433a-822c-dc68297fdd37" alt="traj_bar_chart" width="600"/>
</div>

#### UAV-VLPA* Trajectory
<div align="center">
  <img src="https://github.com/user-attachments/assets/11cc66e9-ec0f-4bd4-baad-a3c24fd4f52a" alt="traj_bar_chart" width="600"/>
</div>

#### Error Analysis
<div align="center">
  <img src="https://github.com/user-attachments/assets/002077a0-5582-4643-b6b4-84f07b1eec77" alt="error_box_plot" width="500"/>
</div>

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{uav-vlpa,
  title={UAV-VLPA*: A Vision-Language-Path-Action System for Optimal Route Generation},
  author={Your Name},
  year={2024}
}
```


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
  <img src="https://github.com/user-attachments/assets/84fb1e8e-4926-4e73-bf7f-fbb83a3fdc33" alt="examples_path_generated2" width="600"/>
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

