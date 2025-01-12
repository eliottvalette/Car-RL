track_out = [
    (25 , 150),    
    (150, 25 ),    
    (375, 25 ),    
    (425, 150),
    (375, 275),
    (425, 325),
    (475, 475),   
    (375, 625),   
    (200, 675),   
    (25 , 675),    
    (75 , 575),
    (125, 525),
    (25 , 375)
]

track_in = [
    (175, 175),   
    (325, 175),   
    (275, 275),
    (275, 425),
    (325, 525),
    (175, 575),   
    (225, 450),
    (175, 375)
]

checkpoints = [
    (150, 135),  
    (365, 155),  
    (355, 565),  
    (135, 575),  
    (100, 375)   
]

new_track_out = []
new_track_in = []
new_checkpoints = []

for point in track_out:
    new_track_out.append((int(point[1] / 700 * 1000), int(point[0] / 700 * 1000)))

for point in track_in:
    new_track_in.append((int(point[1] / 700 * 1000), int(point[0] / 700 * 1000)))

for point in checkpoints:
    new_checkpoints.append((int(point[1] / 700 * 1000), int(point[0] / 700 * 1000)))

print(new_track_out)
print(new_track_in)
print(new_checkpoints)

print(int(0.18426191  * 1000), int(0.45450386 * 800))