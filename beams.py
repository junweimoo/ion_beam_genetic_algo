import random
import numpy as np

class Beam:
    def __init__(self, target, angle):
        if not (0 <= angle <= 360):
            raise ValueError("Angle must be between 0 and 360 degrees")

        self.target = target  # Target should be a tuple of coordinates (x, y)
        self.angle = angle

    def mutate_angle(self, noise_std):
        noise = np.random.normal(0, noise_std, size=1)
        self.angle = self.angle + noise
        if self.angle > 360:
            self.angle -= 360

    def mutate_target(self, noise_std, voxel_matrix):
        noise = np.random.normal(0, noise_std, size=2)
        new_target = np.array(self.target) + noise
        self.target = tuple(np.round(new_target).astype(int))
        if self.target[0] > voxel_matrix.height:
            self.target = (voxel_matrix.height, self.target[1])
        if self.target[1] > voxel_matrix.width:
            self.target = (self.target[0], voxel_matrix.width)

    def __repr__(self):
        return f"Beam(target={self.target}, angle={self.angle} degrees)"


class BeamSet:
    def __init__(self, init_beams = []):
        self.beams = init_beams

    def add_beam(self, target, angle):
        new_beam = Beam(target, angle)
        self.beams.append(new_beam)

    def __repr__(self):
        return "\n".join(str(beam) for beam in self.beams)


# Example usage
beam_set = BeamSet()
beam_set.add_beam((5, 10), 45)
beam_set.add_beam((3, 15), 90)

print(beam_set)
