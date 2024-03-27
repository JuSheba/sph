import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio

def read_coordinates_with_pandas(filename):
    data = {"Time": [], "X": [], "Y": []}
    time_point = None
    all_dfs = []

    with open(filename, "r") as file:
        for line in file:
            if line.startswith("# Time"):
                if time_point is not None:
                    df = pd.DataFrame(data)
                    all_dfs.append(df)
                    data = {"Time": [], "X": [], "Y": []}
                time_point = float(line.split(":")[1].strip())
            else:
                x, y = map(float, line.strip().split())
                data["Time"].append(time_point)
                data["X"].append(x)
                data["Y"].append(y)

        df = pd.DataFrame(data)
        all_dfs.append(df)

    return all_dfs

def count_lines_in_file(filename):
    with open(filename, "r") as file:
        num_lines = sum(1 for line in file)
    return num_lines

def create_gif_from_dataframes(dfs, filename):
    images = []
    if not os.path.exists("pics"):
        os.makedirs("pics")
    
    for i, df in enumerate(dfs):
        plt.figure(figsize=(8, 8))
        plt.scatter(df['X'], df['Y'], color='blue', s=0.5, alpha=0.5)
        plt.title(f"Frame {i+1}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.xlim(-0.6, 0.6)
        plt.ylim(-0.6, 0.6)
        plt.grid(True)
        
        img_path = os.path.join("pics", f"frame_{i}.png")
        plt.savefig(img_path)
        plt.close()
        
        images.append(imageio.imread(img_path))
    
    imageio.mimsave(filename, images, duration=0.5)

filename = "data_test.txt"
dfs = read_coordinates_with_pandas(filename)

gif_filename = "animation.gif"
create_gif_from_dataframes(dfs, gif_filename)
