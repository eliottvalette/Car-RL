### Car Racing Game Environment (race_game.py) ###
import pygame
import math
import numpy as np
import random as rd
from config import Config
import time

config = Config()

class CarRacingGame:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 1000, 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Car Racing RL Environment")

        # Nouvelle palette de couleurs
        self.COLORS = {
            'background': (34, 139, 34),
            'grass': (34, 139, 34),  # Herbe en vert gazon
            'track': (200, 200, 200), # Piste en gris clair   
            'car': (255, 0, 0),
            'checkpoint_active': (255, 215, 0),
            'checkpoint_inactive': (70, 70, 70),
            'text': (0, 0, 0),
            'speed_gauge': (255, 255, 255),
            'speed_needle': (255, 0, 0),
            'laser': (255, 0, 0),      # Couleur des lasers
            'laser_point': (255, 100, 100),  # Couleur des points d'intersection
            'border_red': (255, 0, 0),      # Bordure rouge
            'border_white': (255, 255, 255),  # Bordure blanche
            'info_container': (0, 0, 0, 150)  # Conteneur d'info semi-transparent (RGBA)
        }

        # Track boundaries with wider road
        self.outer_border = config.outer_border_complex
        self.inner_border = config.inner_border_complex
        self.checkpoints = config.checkpoints_complex

        self.current_checkpoint = 0
        self.laps = 0
        self.done = False

        # Amélioration de la voiture
        # Création d'une surface plus grande pour la F1
        self.car_img = pygame.Surface((60, 30), pygame.SRCALPHA)
        
        # Couleurs pour la F1
        f1_main_color = self.COLORS['car']
        arrow_color = (255, 255, 0)  # Jaune vif pour la flèche directionelle
        
        # Corps principal de la F1
        pygame.draw.ellipse(self.car_img, f1_main_color, (15, 5, 30, 20))
        
        # Aileron avant (en fait c'est l'arrière)
        pygame.draw.rect(self.car_img, f1_main_color, (5, 10, 15, 10))
        pygame.draw.rect(self.car_img, f1_main_color, (0, 8, 5, 14))
        
        # Aileron arrière (en fait c'est l'avant)
        pygame.draw.rect(self.car_img, f1_main_color, (45, 10, 10, 10))
        pygame.draw.rect(self.car_img, f1_main_color, (55, 8, 5, 14))
        
        # Flèche directionnelle à l'avant (du côté droit de la voiture)
        pygame.draw.polygon(self.car_img, arrow_color, [(60, 15), (52, 10), (52, 20)])
        
        # Cockpit (réintégré)
        pygame.draw.ellipse(self.car_img, (0, 0, 0), (25, 10, 10, 8))
        
        # Pneus/roues
        pygame.draw.ellipse(self.car_img, (0, 0, 0), (10, 2, 8, 6))   # Arrière gauche
        pygame.draw.ellipse(self.car_img, (0, 0, 0), (10, 22, 8, 6))  # Arrière droite
        pygame.draw.ellipse(self.car_img, (0, 0, 0), (40, 2, 8, 6))   # Avant gauche
        pygame.draw.ellipse(self.car_img, (0, 0, 0), (40, 22, 8, 6))  # Avant droite
        
        # Détails supplémentaires
        pygame.draw.rect(self.car_img, f1_main_color, (22, 5, 16, 20), 1)  # Contour du corps
        pygame.draw.line(self.car_img, f1_main_color, (30, 5), (30, 25), 1)  # Ligne centrale

        self.car_pos = config.complex_spawns[self.current_checkpoint]
        self.car_angle = config.complex_rotations[self.current_checkpoint]
        self.car_speed = 0

        # Rendering
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self.max_steps = 2_000  
        self.steps = 0         
        self.current_reward = 0 

        # Add sensor parameters
        self.num_sensors = 8
        self.sensor_range = 100 
        self.sensor_angles = [i * (360 / self.num_sensors) for i in range(self.num_sensors)]

        # Effet de particules pour les checkpoints
        self.checkpoint_particles = []

    def reset(self):
        self.current_checkpoint = 0
        self.car_pos = config.complex_spawns[self.current_checkpoint]
        self.car_angle = config.complex_rotations[self.current_checkpoint]
        self.car_speed = 0
        self.laps = 0
        self.done = False
        self.steps = 0        
        self.current_reward = 0  
        return self.get_state()

    def step(self, action):
        self.steps += 1  # Add this at start of step function
        prev_distance = np.linalg.norm(np.array(self.checkpoints[self.current_checkpoint]) - np.array(self.car_pos))
        prev_pos = self.car_pos.copy()  # Store previous position
        
        if action == 0:  # Accelerate
            self.car_speed = min(self.car_speed + 0.1, 3)
        elif action == 1:  # Decelerate
            self.car_speed = max(self.car_speed - 0.1, 0)
        elif action == 2:  # Turn Left
            self.car_angle += 3
            # Reduce speed when turning based on current speed
            turn_penalty = 0.05 * self.car_speed
            self.car_speed = max(self.car_speed - turn_penalty, 0.5)
        elif action == 3:  # Turn Right
            self.car_angle -= 3
            # Reduce speed when turning based on current speed
            turn_penalty = 0.15 * self.car_speed
            self.car_speed = max(self.car_speed - turn_penalty, 0.5)

        # Update car position with boundary constraints
        new_pos = [
            self.car_pos[0] + self.car_speed * math.cos(math.radians(self.car_angle)),
            self.car_pos[1] - self.car_speed * math.sin(math.radians(self.car_angle))
        ]

        # Check collision with both track boundaries
        collision = False
        car_rect = pygame.Rect(0, 0, 30, 15)
        car_rect.center = (new_pos[0], new_pos[1])
        
        buffer = 5
        # Check both tracks with different behaviors
        for track_idx, track in enumerate([self.outer_border, self.inner_border]):
            for i in range(len(track)):
                start = track[i]
                end = track[(i + 1) % len(track)]
                
                if car_rect.clipline(
                    (start[0] - buffer, start[1] - buffer),
                    (end[0] + buffer, end[1] + buffer)
                ):
                    if car_rect.clipline(start, end):
                        collision = True
                        # Calculate wall direction vector
                        wall_dx = end[0] - start[0]
                        wall_dy = end[1] - start[1]
                        wall_len = math.sqrt(wall_dx**2 + wall_dy**2)
                        wall_dx /= wall_len
                        wall_dy /= wall_len

                        # Calculate wall normal (perpendicular to wall)
                        if track_idx == 0: 
                            wall_normal_x = -wall_dy
                            wall_normal_y = wall_dx
                        else: 
                            wall_normal_x = wall_dy
                            wall_normal_y = -wall_dx

                        # Stronger push when hitting walls
                        push_distance = 2  # Increased from 1 to 2
                        new_pos[0] += wall_normal_x * push_distance
                        new_pos[1] += wall_normal_y * push_distance

                        # More speed reduction on collision
                        self.car_speed *= 0.5  # Increased speed reduction from 0.6 to 0.5
                        break

        self.car_pos = new_pos  # Use the calculated position with wall push

        # Check checkpoints with larger collision box
        checkpoint_rect = pygame.Rect(0, 0, 80, 80)  # Increased from 20x20 to 40x40
        checkpoint_rect.center = self.checkpoints[self.current_checkpoint]
        if checkpoint_rect.collidepoint(self.car_pos):
            self.current_checkpoint = (self.current_checkpoint + 1) % len(self.checkpoints)
            if self.current_checkpoint == 0:
                self.laps += 1

        # Modify done condition to include max steps
        self.done = self.steps >= self.max_steps

        # Reward management
        reward = 0
        
        # 1. Distance-based reward
        new_distance = np.linalg.norm(np.array(self.checkpoints[self.current_checkpoint]) - np.array(self.car_pos))

        if new_distance < prev_distance:
            reward += 0.2
        else :
            reward -= 0.2

        # 2. Checkpoint rewards
        if checkpoint_rect.collidepoint(self.car_pos):
            reward += 150

        # 3. Speed-based continuous reward
        speed_reward = (self.car_speed - 1.0) * 0.6
        reward += speed_reward

        # 4. Penalties
        # Collision penalty
        if collision:
            reward -= 4  # Smaller penalty for hitting walls

        # After calculating final reward, store it
        self.current_reward += reward
        
        return self.get_state(), reward, self.done, {}
    
    def get_wall_distances(self):
        """Calculate distances to walls in 8 directions around the car"""
        distances = []
        
        for angle in self.sensor_angles:
            # Calculate sensor endpoint using car's position and angle
            sensor_angle = math.radians(self.car_angle + angle)
            end_x = self.car_pos[0] + math.cos(sensor_angle) * self.sensor_range
            end_y = self.car_pos[1] - math.sin(sensor_angle) * self.sensor_range
            
            min_distance = self.sensor_range  # Default to max range
            
            # Check both tracks
            for track in [self.outer_border, self.inner_border]:
                for i in range(len(track)):
                    start_wall = track[i]
                    end_wall = track[(i + 1) % len(track)]
                    
                    # Line intersection calculation
                    x1, y1 = self.car_pos
                    x2, y2 = end_x, end_y
                    x3, y3 = start_wall
                    x4, y4 = end_wall
                    
                    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                    if denominator == 0:
                        continue                        
                    
                    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
                    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
                    
                    if 0 <= t <= 1 and 0 <= u <= 1:  # Lines intersect
                        # Calculate intersection point
                        intersection_x = x1 + t * (x2 - x1)
                        intersection_y = y1 + t * (y2 - y1)
                        
                        # Calculate distance to intersection
                        distance = math.sqrt(
                            (intersection_x - x1) ** 2 + 
                            (intersection_y - y1) ** 2
                        )
                        min_distance = min(min_distance, distance)
            
            # Normalize distance to [0, 1]
            distances.append(min_distance / self.sensor_range)
            
        return distances
    
    def get_state(self):
        # Get previous state components
        car_pos = np.array(self.car_pos) / np.array([self.WIDTH, self.HEIGHT])
        car_speed = np.array([self.car_speed * math.cos(math.radians(self.car_angle)), 
                             self.car_speed * math.sin(math.radians(self.car_angle))])
        car_angle = np.array([math.cos(math.radians(self.car_angle)), 
                             math.sin(math.radians(self.car_angle))])
        
        checkpoint = np.array(self.checkpoints[self.current_checkpoint]) / np.array([self.WIDTH, self.HEIGHT])
        checkpoint_direction = checkpoint - car_pos 
        checkpoint_distance = [np.linalg.norm(checkpoint_direction)]
        checkpoint_one_hot = np.zeros(len(self.checkpoints))
        checkpoint_one_hot[self.current_checkpoint] = 1
        
        # Calculate relative angle to checkpoint
        car_direction = np.array([math.cos(math.radians(self.car_angle)), 
                                -math.sin(math.radians(self.car_angle))])
        to_checkpoint = checkpoint - car_pos
        to_checkpoint = to_checkpoint / np.linalg.norm(to_checkpoint)
        
        dot_product = np.dot(car_direction, to_checkpoint)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        cross_product = np.cross(car_direction, to_checkpoint)
        
        angle = math.acos(dot_product)
        if cross_product < 0:
            angle = -angle
        normalized_relative_angle = angle / math.pi
        
        # Get wall distances
        wall_distances = np.array(self.get_wall_distances())
        
        # Concatenate all state components including wall distances
        state = np.concatenate([
            car_pos,                    # 2 values
            car_speed,                  # 2 values
            car_angle,                  # 2 values
            checkpoint,                 # 2 values
            to_checkpoint,              # 2 values
            checkpoint_distance,        # 1 value
            checkpoint_one_hot,         # 5 values
            [normalized_relative_angle], # 1 value
            wall_distances              # 8 values (new)
        ])
        
        return state

    def draw_striped_border(self, points, width=10):
        """Dessine une bordure rayée rouge et blanche autour du circuit"""
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            
            # Calculer la longueur et l'angle du segment
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.sqrt(dx*dx + dy*dy)
            angle = math.atan2(dy, dx)
            
            # Nombre de rayures (chaque rayure = 20 pixels)
            stripe_length = 20
            num_stripes = max(2, int(length / stripe_length))
            actual_stripe_length = length / num_stripes
            
            # Dessiner chaque rayure alternée rouge et blanche
            for j in range(num_stripes):
                # Calculer les points de début et de fin de cette rayure
                stripe_start_x = start[0] + j * actual_stripe_length * math.cos(angle)
                stripe_start_y = start[1] + j * actual_stripe_length * math.sin(angle)
                stripe_end_x = start[0] + (j + 1) * actual_stripe_length * math.cos(angle)
                stripe_end_y = start[1] + (j + 1) * actual_stripe_length * math.sin(angle)
                
                # Alterner rouge et blanc
                color = self.COLORS['border_red'] if j % 2 == 0 else self.COLORS['border_white']
                
                # Dessiner la rayure
                pygame.draw.line(
                    self.screen, 
                    color, 
                    (stripe_start_x, stripe_start_y), 
                    (stripe_end_x, stripe_end_y), 
                    width
                )

    def render(self, fps=60):
        self.screen.fill(self.COLORS['background'])

        # Dessiner l'herbe (extérieur)
        pygame.draw.polygon(self.screen, self.COLORS['track'], self.outer_border)
        # Dessiner la piste (intérieur)
        pygame.draw.polygon(self.screen, self.COLORS['grass'], self.inner_border)
        
        # Dessiner les bordures rayées rouge et blanc
        self.draw_striped_border(self.outer_border, width=8)
        self.draw_striped_border(self.inner_border, width=8)

        # Dessiner les checkpoints
        for i, cp in enumerate(self.checkpoints):
            color = self.COLORS['checkpoint_active'] if i == self.current_checkpoint else self.COLORS['checkpoint_inactive']
            pygame.draw.circle(self.screen, color, cp, 15)
            pygame.draw.circle(self.screen, (255, 255, 255), cp, 10)

        # Dessiner la voiture
        rotated_car = pygame.transform.rotate(self.car_img, self.car_angle)
        car_rect = rotated_car.get_rect(center=self.car_pos)
        self.screen.blit(rotated_car, car_rect)

        # Dessiner l'interface utilisateur améliorée
        # Jauge de vitesse
        speed_gauge_pos = (self.WIDTH - 100, 50)
        pygame.draw.circle(self.screen, self.COLORS['speed_gauge'], speed_gauge_pos, 30)
        speed_angle = (self.car_speed / 3) * 180  # Normaliser la vitesse
        end_x = speed_gauge_pos[0] + 25 * math.cos(math.radians(speed_angle - 90))
        end_y = speed_gauge_pos[1] + 25 * math.sin(math.radians(speed_angle - 90))
        pygame.draw.line(self.screen, self.COLORS['speed_needle'], speed_gauge_pos, (end_x, end_y), 3)

        # Créer un conteneur semi-transparent pour les informations
        info_surface = pygame.Surface((200, 130), pygame.SRCALPHA)
        # Dessiner un rectangle arrondi semi-transparent
        pygame.draw.rect(info_surface, self.COLORS['info_container'], (0, 0, 200, 130), border_radius=10)
        # Afficher le conteneur d'informations
        self.screen.blit(info_surface, (5, 5))

        # Informations de jeu
        speed_text = self.font.render(f'Speed: {self.car_speed:.1f}', True, self.COLORS['border_white'])
        lap_text = self.font.render(f'Laps: {self.laps}', True, self.COLORS['border_white'])
        reward_text = self.font.render(f'Reward: {self.current_reward:.0f}', True, self.COLORS['border_white'])
        
        self.screen.blit(speed_text, (20, 20))
        self.screen.blit(lap_text, (20, 60))
        self.screen.blit(reward_text, (20, 100))

        """
        # Dessiner les capteurs
        for angle, distance in zip(self.sensor_angles, self.get_wall_distances()):
            sensor_angle = math.radians(self.car_angle + angle)
            end_x = self.car_pos[0] + math.cos(sensor_angle) * (distance * self.sensor_range)
            end_y = self.car_pos[1] - math.sin(sensor_angle) * (distance * self.sensor_range)
            
            # Capteurs plus visibles avec effet de brillance
            # Ligne principale
            pygame.draw.line(self.screen, self.COLORS['laser'], self.car_pos, (end_x, end_y), 3)
            # Effet de brillance
            pygame.draw.line(self.screen, (255, 200, 200), self.car_pos, (end_x, end_y), 1)
            
            # Point d'intersection plus visible
            pygame.draw.circle(self.screen, self.COLORS['laser_point'], (int(end_x), int(end_y)), 5)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(end_x), int(end_y)), 2)
        """

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        pygame.quit()

    def manual_control(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            # Get keyboard input and process actions
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.step(0)
            elif keys[pygame.K_DOWN]:
                self.step(1)
            if keys[pygame.K_LEFT]:
                self.step(2)
            elif keys[pygame.K_RIGHT]:
                self.step(3)

            if keys[pygame.K_r]:
                self.reset()
                time.sleep(0.5)

            if keys[pygame.K_ESCAPE]:
                print("Decomposed State")
                state = self.get_state()
                print("Car Position:", state[:2])
                print("Car Speed:", state[2:4])
                print("Car Angle:", state[4:6])
                print("Checkpoint Position:", state[6:8])
                print("Checkpoint Direction:", state[8:10])
                print("Checkpoint Distance:", state[10])
                print("Relative Angle:", state[11])
                print("Checkpoint One-Hot:", state[12:17])
                print("Wall Distances:", state[17:])

            # Use the render method instead of duplicating rendering code
            self.render(60)

if __name__ == "__main__":
    game = CarRacingGame()
    game.manual_control()
    game.close()

