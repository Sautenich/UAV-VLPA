import re

def extract_home_positions(home_position_data):
    home_positions = {}
    for line in home_position_data.splitlines():
        if 'Image' in line:
            img_num = int(re.search(r'Image (\d+)', line).group(1))
            home_positions[img_num] = {}
        elif 'Latitude' in line:
            home_positions[img_num]['Latitude'] = float(re.search(r'Latitude = ([\d\-.]+)', line).group(1))
        elif 'Longitude' in line:
            home_positions[img_num]['Longitude'] = float(re.search(r'Longitude = ([\d\-.]+)', line).group(1))
    return home_positions

def add_home_positions_to_results(results_file, home_position_file, output_file):
    # Read result coordinates file
    with open(results_file, 'r') as file:
        result_data = file.read()

    # Read home positions file
    with open(home_position_file, 'r') as file:
        home_positions = extract_home_positions(file.read())

    # Initialize output data
    updated_result_data = result_data

    # Add home positions to the result file
    for img_num, position in home_positions.items():
        # Prepare the home position info
        home_position_str = (
            f"  Home Position:\n"
            f"    Latitude: {position['Latitude']}\n"
            f"    Longitude: {position['Longitude']}"
        )

        # Regex to find the image number section and insert home position after the image number
        img_section_pattern = rf"(Image {img_num}:)"
        
        # Insert home position after the image number
        updated_result_data = re.sub(
            img_section_pattern,
            rf"\1\n{home_position_str}",
            updated_result_data,
            count=1
        )

    # Write the updated result data to the output file
    with open(output_file, 'w') as file:
        file.write(updated_result_data)

    print(f"Home positions added and saved to {output_file}")

# # Example usage
# results_file = '../result_coordinates.txt'
# home_position_file = 'home_position.txt'
# output_file = 'VLM_coordinates.txt'

# add_home_positions_to_results(results_file, home_position_file, output_file)
