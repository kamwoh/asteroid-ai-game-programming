import numpy as np

from ai.sensor import sense_eight_dir
from asteroids.player import Player
from asteroids.utils import LINEAR


class AI_PlayerRL(Player):
    """
    Defines the player ship, controlled by an AI.
    """

    # Size of the decision vector
    DECISION_VECTOR_SIZE = 4

    def __init__(self, x, y):
        super(AI_PlayerRL, self).__init__(x, y)

    def sense(self, asteroids, bullets):
        """
        Checks the state of the world, and returns a feature
        matrix to be used as input to the AI update function.
        """

        directions = sense_eight_dir(self, asteroids, 300, shape=LINEAR)
        speed = self.speed
        rotation = self.rotation
        # directions.append(speed / Player.MAX_SPEED)
        # directions.append(rotation / (2 * math.pi))
        return np.array(directions)

    def update(self, bullets, sensor_data):
        """
        Updates any time dependent player state, then runs
        the AI algorithm on sensor_data, and performs the
        appropiate actions in response.
        """
        super(AI_PlayerRL, self).update(bullets, sensor_data)

    def perform_decisions(self, decision_vector, bullets):
        """
        Accepts a boolean vector containing the following decisions:
          0: Whether to shoot
          1: Whether to boost
          2: Whether to spin clockwise
          3: Whether to spin counter-clockwise

        The player ship then carries out these decisions for this timestep.
        """
        if len(decision_vector) != AI_PlayerRL.DECISION_VECTOR_SIZE:
            raise RuntimeError(("Programmer Error: decision vector has "
                                "length '%d' instead of expected length '%d'." %
                                (len(decision_vector), AI_PlayerRL.DECISION_VECTOR_SIZE)))

        # if decision_vector[0]:
        self.shoot(bullets)

        if decision_vector[1]:
            self.start_boosting()
        else:
            self.stop_boosting()

        # Spin in the decided upon direction, or stop spinning entirely
        # Note: if both spin decisions are True, arbitrarily spin clockwise
        if decision_vector[2]:
            self.start_spinning(True)
        elif decision_vector[3]:
            self.start_spinning(False)
        else:
            self.stop_spinning()
