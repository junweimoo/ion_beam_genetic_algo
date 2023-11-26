class Voxel:
    def __init__(self, voxel_type, target):
        self.voxel_type = voxel_type # PV, OA, or NT
        self.target = target

    def __repr__(self):
        return str(self.target)
        return f"Voxel(type={self.voxel_type}, target={self.target})"


class VoxelMatrix:
    def __init__(self):
        self.matrix = []
        self.width = 0
        self.heght = 0

    def read_from_file(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                row = []
                entries = line.strip().split(',')
                for entry in entries:
                    voxel_type, target = entry.split()
                    target = float(target)
                    if voxel_type in {'PV', 'NT', 'OA'}:
                        row.append(Voxel(voxel_type, target))
                    else:
                        raise ValueError("Invalid voxel type or target value")
                self.matrix.append(row)
            self.width = len(self.matrix[0])
            self.height = len(self.matrix)

    def __repr__(self):
        return "\n".join([" ".join([str(voxel) for voxel in row]) for row in self.matrix])


# Example usage
voxel_matrix = VoxelMatrix()
voxel_matrix.read_from_file('voxel_data_2.txt')
print(voxel_matrix)

