from dataclasses import dataclass

@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def __str__(self) -> str:
        return f'''({self.x:8.2f}, {self.y:8.2f}, {self.z:8.2f})'''


