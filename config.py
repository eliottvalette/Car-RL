class Config():
    def __init__(self):
        self.outer_track_complex = [
            # Top-Left Corner (1)
            (50, 50),

            # ZigZag part 2 -> 8
            (200, 50),
            (300, 150),
            (400, 50),
            (500, 150),
            (600, 50),
            (700, 150),
            (800, 50),

            # Top-Right Corner (9)
            (950, 50),

            # Bottom-Right Corner (10)
            (950, 750),

            # Spiral part 11 -> 16
            (500, 750),
            (500, 500),
            (700, 500),
            (700, 400),
            (400, 400),
            (400, 750),

            # Bottom-Left Corner (17)
            (50, 750)
        ]

        self.inner_track_complex = [
            # Top-Left Corner (1)
            (150, 150),

            # ZigZag part 2 -> 8
            (200, 150),
            (300, 250),
            (400, 150),
            (500, 250),
            (600, 150),
            (700, 250),
            (800, 150),

            # Top-Right Corner (9)
            (850, 150),

            # Bottom-Right Corner (10)
            (850, 650),

            # Spiral part 11 -> 16
            (600, 650),
            (600, 600),
            (800, 600),
            (800, 300),
            (300, 300),
            (300, 650),

            # Bottom-Left Corner (17)
            (150, 650)

        ]

        self.checkpoints_complex = [
            (125, 100),    
            (875, 100),    
            (875, 700), 
            (550, 650),
            (750, 500),
            (550, 350),
            (125, 700)     
        ]

        self.complex_spawns = [
            [100, 450],
            [500, 200],
            [900, 500],
            [700, 700],
            [600, 550],
            [750, 550],
            [350, 450]
        ]

        self.complex_rotations = [
            90,
            0,
            270,
            180,
            0,
            90,
            270
        ]

        self.outer_track_simple = [
            # Top-Left Corner (1)
            (50, 50),
            # Top-Right Corner (2)
            (950, 50),
            # Bottom-Right Corner (3)
            (950, 750),
            # Bottom-Left Corner (4)
            (50, 750)
        ]

        self.inner_track_simple = [
            # Top-Left Corner (1)
            (150, 150),
            # Top-Right Corner (2)
            (850, 150),
            # Bottom-Right Corner (3)
            (850, 650),
            # Bottom-Left Corner (4)
            (150, 650)
        ]

        self.checkpoints_simple = [
            (125, 100),    
            (875, 100),    
            (875, 700),    
            (500, 700),
            (125, 700)     
        ] 

        self.simple_spawns = [
            [100, 450],
            [500, 100],
            [900, 450],
            [500, 700],
            [350, 450]
        ]

        self.simple_rotations = [
            90,
            0,
            180,
            270,
            180
        ]