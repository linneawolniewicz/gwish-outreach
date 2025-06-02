import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import ipywidgets as widgets
import matplotlib.patches as patches
from craterpy import CraterDatabase
from PIL import Image
from IPython.display import display

def plot_full_moon(full_moon, show_boxes, areas, colormap, min_lon, max_lon, min_lat, max_lat):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Lunar Surface with Marked Areas")
    
    # Display the image
    ax.imshow(full_moon, cmap=colormap)
    
    # Label axes with lat/lon values
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(np.linspace(0, full_moon.width, 5))
    ax.set_xticklabels(np.round(np.linspace(min_lon, max_lon, 5), decimals=0))
    ax.set_yticks(np.linspace(0, full_moon.height, 5))
    ax.set_yticklabels(np.round(np.linspace(max_lat, min_lat, 5), decimals=0))
    
    # Function to convert lat/lon to image pixel coordinates
    def latlon_to_pixels(lat, lon, img_width, img_height):
        x = ((lon - min_lon) / (max_lon - min_lon)) * img_width
        y = ((max_lat - lat) / (max_lat - min_lat)) * img_height
        return x, y
    
    # Add blue rectangles for each area and label them if enabled
    if show_boxes:
        for _, row in areas.iterrows():
            ul_x, ul_y = latlon_to_pixels(row['upper_left_lat'], row['upper_left_lon'], full_moon.width, full_moon.height)
            lr_x, lr_y = latlon_to_pixels(row['lower_right_lat'], row['lower_right_lon'], full_moon.width, full_moon.height)
            
            width = lr_x - ul_x
            height = lr_y - ul_y
            rect = patches.Rectangle((ul_x, ul_y), width, height, linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            
            # Label the area with its number
            ax.text(ul_x + width / 2, ul_y + height / 2, str(int(row['area_number'])), color='white', fontsize=12, ha='center', va='center')
    
    plt.show()

def plot_area_craters(min_diameter, area_number, area_path, colormap, area_craters, min_lon, max_lon, min_lat, max_lat):
    area_craters = area_craters.loc[(area_craters.DIAM_CIRC_IMG > min_diameter)]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5)) 

    # Plot the image with craters
    if not area_craters.empty:
        area_crater_counts = CraterDatabase(area_craters, units="km")
        area_crater_counts.add_annuli(0, 1, 'crater')
        ax = area_crater_counts.plot(ax=ax, lw=0.75, alpha=1, color='white')

    # Plot the image of the area
    im = plt.imread(area_path)
    ax.set_title(f"Lunar Surface of Area {area_number}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if (min_lon > 180) & (max_lon > 180):
        ax.imshow(im, extent=[min_lon - 360, max_lon - 360, min_lat, max_lat], cmap=colormap)
    else:
        ax.imshow(im, extent=[min_lon, max_lon, min_lat, max_lat], cmap=colormap)
    plt.show()

    print(f"Number of craters with diameter greater than {min_diameter} km: {area_craters.shape[0]}")

    # Print the type of area
    if area_number in [1, 3, 5, 7]:
        print("This area is a highlands area.")
    elif area_number in [2, 4, 6, 8]:
        print("This area is a mare area.")
    else:
        raise ValueError("Invalid area number. Please choose a number between 1 and 8.")
        
def plot_age_lines(mare_crater_counts, highlands_crater_counts, diameters, reference_area, image_path, mare_color, highlands_color):
    # Mare area
    mare_cumulative_count_normalized = np.array(mare_crater_counts) / reference_area
    mare_mask = np.array(mare_cumulative_count_normalized) > 0
    mare_diameters = np.array(diameters)[mare_mask]
    mare_cumulative_count_normalized = np.array(mare_cumulative_count_normalized)[mare_mask]

    # Highlands area
    highlands_cumulative_count_normalized = np.array(highlands_crater_counts) / reference_area
    highlands_mask = np.array(highlands_cumulative_count_normalized) > 0
    highlands_diameters = np.array(diameters)[highlands_mask]
    highlands_cumulative_count_normalized = np.array(highlands_cumulative_count_normalized)[highlands_mask]

    # Load image
    img = np.asarray(Image.open(image_path))

    # Convert pixel positions to log-scale data values
    img_height, img_width = img.shape[:2]
    x_min, x_max = 10**-0.4, 10**1.6
    y_min, y_max = 1e-4, 1e-2
    x_ticks = [1, 10]  # Only 1 km and 10 km
    y_ticks = [1e-4, 1e-3, 1e-2]  # Only 10^-4, 10^-3, 10^-2
    x_pixel_positions = np.interp(np.log10(x_ticks), [np.log10(x_min), np.log10(x_max)], [0, img_width])
    y_pixel_positions = np.interp(np.log10(y_ticks), [np.log10(y_min), np.log10(y_max)], [img_height, 0])

    # Plot the image
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray", origin="upper")

    # Prepare the mare data
    log_mare_diameters = np.log10(mare_diameters)
    mare_diameter_pixel_positions = np.interp(log_mare_diameters, [np.log10(x_min), np.log10(x_max)], [0, img_width])
    log_mare_cumulative_counts = np.log10(mare_cumulative_count_normalized)
    mare_density_pixel_positions = np.interp(log_mare_cumulative_counts, [np.log10(y_min), np.log10(y_max)], [img_height, 0])

    # Prepare the highlands data
    log_highlands_diameters = np.log10(highlands_diameters)
    highlands_diameter_pixel_positions = np.interp(log_highlands_diameters, [np.log10(x_min), np.log10(x_max)], [0, img_width])
    log_highlands_cumulative_counts = np.log10(highlands_cumulative_count_normalized)
    highlands_density_pixel_positions = np.interp(log_highlands_cumulative_counts, [np.log10(y_min), np.log10(y_max)], [img_height, 0])

    # Plot the mare and highlands data
    ax.plot(mare_diameter_pixel_positions, mare_density_pixel_positions, color=mare_color, marker='o', label="Mare Crater Data")
    ax.plot(highlands_diameter_pixel_positions, highlands_density_pixel_positions, color=highlands_color, marker='o', label="Highlands Crater Data")

    # Set custom x and y axes
    ax.set_xticks(x_pixel_positions)
    ax.set_xticklabels([f"{xt:.0f} km" for xt in x_ticks])
    ax.set_yticks(y_pixel_positions)
    ax.set_yticklabels([f"$10^{{{int(np.log10(yt))}}}$" for yt in y_ticks]) 
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    ax.set_xlabel("Diameter (km)")
    ax.set_ylabel("Cumulative Crater Density (km$^{-2}$)")
    ax.legend(loc="lower left")

    return fig, ax