def coordinates_from_json(json_data, width, height):
    coordinates_list = []
    for object, data in json_data.items():
        # Extract coordinates
        coordinates = data['coordinates']
        x_percent = coordinates[0]
        y_percent = coordinates[1]

        # Convert percentage to pixel values
        x_pixel = (x_percent / 100) * width
        y_pixel = (y_percent / 100) * height
        coordinates_list.append([x_pixel, y_pixel])
    return coordinates_list