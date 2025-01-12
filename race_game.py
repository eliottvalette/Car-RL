### Car Racing Game Environment (race_game.py) ###
import pygame
import math
import numpy as np
import random as rd

class CarRacingGame:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 1000, 800
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Car Racing RL Environment")

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        # Car properties
        self.car_img = pygame.Surface((40, 20), pygame.SRCALPHA)
        pygame.draw.polygon(self.car_img, self.RED, [(0, 10), (40, 0), (40, 20)])
        self.car_pos = [80, 150]
        self.car_angle = 0
        self.car_speed = 0

        # Track boundaries with wider road
        self.outer_track = [
            (25, 150),    # Widened outward
            (150, 25),    # Widened outward
            (375, 25),    # Widened outward
            (425, 150),
            (375, 275),
            (425, 325),
            (475, 475),   # Widened outward
            (375, 625),   # Widened outward
            (200, 675),   # Widened outward
            (25, 675),    # Widened outward
            (75, 575),
            (125, 525),
            (25, 375)
        ]
        
        self.inner_track = [
            (175, 175),   # Moved inward
            (325, 175),   # Moved inward
            (275, 275),
            (275, 425),
            (325, 525),
            (175, 575),   # Moved inward
            (225, 450),
            (175, 375)
        ]

        # Adjusted checkpoints to be centered in the track
        self.checkpoints = [
            (150, 135),   # Start straight
            (365, 155),   # Top right corner
            (355, 565),   # Right side
            (135, 575),    # Bottom left
            (100, 375)    # Left side
        ]

        self.current_checkpoint = 0
        self.laps = 0
        self.done = False

        # Rendering
        self.clock = pygame.time.Clock()

        self.max_steps = 2_000  
        self.steps = 0         
        self.current_reward = 0  

    def reset(self):
        self.car_pos = [80, 150]
        self.car_angle = 0
        self.car_speed = 0
        self.current_checkpoint = 0
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
        elif action == 3:  # Turn Right
            self.car_angle -= 3

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
        for track_idx, track in enumerate([self.outer_track, self.inner_track]):
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
                        # For outer track (idx=0): push inward
                        # For inner track (idx=1): push outward
                        if track_idx == 0:  # Outer track
                            wall_normal_x = -wall_dy
                            wall_normal_y = wall_dx
                        else:  # Inner track
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
            speed_bonus = min(self.car_speed, 5) * 0.4
            reward += speed_bonus
        else :
            speed_malus = min(self.car_speed, 5) * 0.4
            reward -= speed_malus

        # 2. Checkpoint rewards
        if checkpoint_rect.collidepoint(self.car_pos):
            base_checkpoint_reward = 150
            # Additional reward for maintaining speed through checkpoint
            speed_bonus = min(self.car_speed, 5) * 2
            reward += base_checkpoint_reward + speed_bonus

        # 3. Lap completion reward
        if self.current_checkpoint == 0 and checkpoint_rect.collidepoint(self.car_pos):
            reward += 50 * (self.laps + 1)  # Increasing reward for each lap

        # 4. Penalties
        # Collision penalty
        if collision:
            reward -= 5  # Smaller penalty for hitting walls

        # After calculating final reward, store it
        self.current_reward += reward
        
        return self.get_state(), reward, self.done, {}

    def get_state(self):
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

        # Properly calculate the relative angle between car's front and checkpoint
        # 1. Get car's forward direction vector (already normalized)
        car_direction = np.array([math.cos(math.radians(self.car_angle)), 
                                -math.sin(math.radians(self.car_angle))])
        
        # 2. Get direction to checkpoint (normalize it)
        to_checkpoint = checkpoint - car_pos
        to_checkpoint = to_checkpoint / np.linalg.norm(to_checkpoint)
        
        # 3. Calculate angle using dot product and cross product
        dot_product = np.dot(car_direction, to_checkpoint)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Use cross product to determine direction (right or left)
        cross_product = np.cross(car_direction, to_checkpoint)
        
        # Calculate angle and normalize to [-1, 1]
        angle = math.acos(dot_product)
        if cross_product < 0:
            angle = -angle
        normalized_relative_angle = angle / math.pi

        # Concatenate all state components
        concate_all_state = np.concatenate([car_pos, car_speed, car_angle, checkpoint, 
                                          to_checkpoint, checkpoint_distance, checkpoint_one_hot,
                                          [normalized_relative_angle]])
        return concate_all_state

    def render(self, fps=60):
        self.screen.fill(self.WHITE)

        # Draw both track boundaries
        pygame.draw.lines(self.screen, self.BLACK, True, self.outer_track, 2)
        pygame.draw.lines(self.screen, self.BLACK, True, self.inner_track, 2)

        # Fill track area
        track_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(track_surface, (200, 200, 200, 128), self.outer_track)
        pygame.draw.polygon(track_surface, (255, 255, 255, 255), self.inner_track)
        self.screen.blit(track_surface, (0, 0))

        # Draw checkpoints with visible collision boxes
        for i, cp in enumerate(self.checkpoints):
            color = self.GREEN if i == self.current_checkpoint else self.BLUE
            # Draw checkpoint circle
            pygame.draw.circle(self.screen, color, cp, 10)
            # Draw checkpoint collision box
            checkpoint_rect = pygame.Rect(0, 0, 40, 40)
            checkpoint_rect.center = cp
            pygame.draw.rect(self.screen, color, checkpoint_rect, 1)

        # Draw the car
        rotated_car = pygame.transform.rotate(self.car_img, self.car_angle)
        car_rect = rotated_car.get_rect(center=self.car_pos)
        self.screen.blit(rotated_car, car_rect)
        
        # Draw UI text
        font = pygame.font.Font(None, 36)
        speed_text = font.render(f'Speed: {self.car_speed:.1f}', True, self.BLACK)
        lap_text = font.render(f'Laps: {self.laps}', True, self.BLACK)
        reward_text = font.render(f'Reward: {self.current_reward:.2f}', True, self.BLACK)
        self.screen.blit(speed_text, (10, 10))
        self.screen.blit(lap_text, (10, 50))
        self.screen.blit(reward_text, (200, 10))
        
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

            # Use the render method instead of duplicating rendering code
            self.render(60)

if __name__ == "__main__":
    game = CarRacingGame()
    game.manual_control()
    game.close()

