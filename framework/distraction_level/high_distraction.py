import numpy as np
from world import World
from agents import Car
from geometry import Point
import time
import math
import threading

import tkinter as tk
from PIL import Image, ImageTk

global autonomous_takeover
global risk_score
global autonomous_takeover_complete
risk_score = 0
PATH_TO_IMAGE = "C:/UWaterloo/Courses/ME 744 - Computational Intelligence/distracted-driving-system/framework/distraction_level/cityscape.png"

def start_simulator():
    # Define world then render
    dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
    width = 220
    height = 120
    ppm = 6
    world = World(dt, width = width, height = height, ppm = ppm) # The world is 120 meters by 120 meters. ppm is the pixels per meter.
    world.render()

    # Display map
    image = Image.open(PATH_TO_IMAGE)
    photo = ImageTk.PhotoImage(image)
    canvas = world.visualizer.win
    canvas.photo_ref = photo
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    x, y = 10, 10
    # Draw rectangle outline (no fill, just border)
    bg_rect = canvas.create_rectangle(
        5, 5, x+290, y+60,  # adjust size to fit text
        fill="light gray", outline="gray", width=5
    )

    # Add text overlay for autonomous mode
    autonomous_mode_text = canvas.create_text(
        x+5, y+5,  # position (x, y) in pixels
        anchor=tk.NW,
        text="Autonomous: OFF",
        fill="red",
        font=("Helvetica", 24, "bold")
    )

    # Define waypoints
    road_waypoints = np.array([[86, 30], # Main
                            [39, 30], # Main
                            [34, 31],
                            [31, 34],
                            [30, 39], # Main
                            [30, 96], # Main
                            [31, 101],
                            [34, 104],
                            [39, 105], # Main
                            [128, 105], # Main
                            [196, 105], # Main
                            [201, 104],
                            [204, 100],
                            [204, 96],  # Main
                            [204, 39],  # Main
                            [204, 35],
                            [201, 31],
                            [196, 30]]) # Main
    road_waypoints[:, 1] = height - road_waypoints[:, 1]

    parking_waypoints = {0: np.array([[86, 30],  # Main
                                    [80, 31],
                                    [77, 35],
                                    [77, 39],  # Main
                                    [77, 46],
                                    [74, 51],
                                    [69, 54],  # Main
                                    [66, 54]]), # Main

                        9: np.array([[128, 105], # Main
                                    [133, 103],
                                    [136, 100],
                                    [137, 96],  # Main
                                    [137, 92],
                                    [139, 88],
                                    [143, 87],  # Main
                                    [ 146, 87]])} # Main

    for key in parking_waypoints:
        parking_waypoints[key][:, 1] = height - parking_waypoints[key][:, 1]

    # Instantiate car and set initial conditions
    car = Car(Point(road_waypoints[-1, 0], road_waypoints[-1, 1]), np.pi)
    car.velocity = Point(10, 0)
    alpha = 0.5
    gamma = 0.06
    world.add(car)
    world.render()

    # Start simulation allowing car to follow waypoints
    iterations = 1000
    distance_threshold = 1
    done = False
    global autonomous_takeover, risk_score, autonomous_takeover_complete
    autonomous_takeover = False
    autonomous_takeover_complete = False

    def move_to(waypoint):
        time_elapsed = 0
        while True:
            car_position = np.array([car.center.x, car.center.y])
            distance_error = np.linalg.norm(car_position - waypoint)
            if distance_error < distance_threshold:
                break

            heading_setpoint = math.atan2(waypoint[1] - car_position[1], waypoint[0] - car_position[0])
            heading_error = (heading_setpoint - car.heading + np.pi) % (2*np.pi) - np.pi
            car.heading += alpha * heading_error

            random_noise = np.random.uniform(-gamma, gamma)
            sin_noise = 0.05 * np.sin(2 * np.pi * 0.3 * time_elapsed)
            global autonomous_takeover, risk_score
            car.heading += (not autonomous_takeover) * (random_noise + sin_noise) * risk_score

            time_elapsed += dt
            world.tick()
            world.render()
            time.sleep(dt/4)

    def running():
        global autonomous_takeover_complete
        # While moving towards the waypoint adjust the heading overtime
        for i in range(iterations):
            if not autonomous_takeover_complete:
                for index, waypoint in enumerate(road_waypoints):
                    car.velocity = Point(10, 0)

                    # if index == 0 and i == 1:
                    #     autonomous_takeover = True
                    if autonomous_takeover:
                        canvas.itemconfig(autonomous_mode_text, text="Autonomous: ON", fill="green")

                    if autonomous_takeover and index in parking_waypoints.keys():
                        for parking_waypoint in parking_waypoints[index]:
                            move_to(parking_waypoint)
                        autonomous_takeover_complete = True
                        car.velocity = Point(0, 0)
                        break

                    move_to(waypoint)
            else:
                world.tick()
                world.render()
                time.sleep(dt/4)
    #running()
    thread = threading.Thread(target=running, daemon=True)
    thread.start()
    #canvas.master.mainloop()
    canvas.master.geometry(f"{width*ppm}x{height*ppm}+{2550-width*ppm}+0")
    #print(width*ppm-2560)
    return canvas.master


def activate_autonomous_takeover(current_risk_score):
    global autonomous_takeover, risk_score
    autonomous_takeover = True
    risk_score = current_risk_score