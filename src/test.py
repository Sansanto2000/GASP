import numpy as np

from gsssp.observationArtist import (
    define_observations_limits,
    visualize_observations,
)

rng = np.random.default_rng(42)

while True:
    observation_limits = define_observations_limits(2048, 2048, rng)

    print(observation_limits)

    visualize_observations(
        observation_limits,
        2048,
        2048,
    )

    key = input("Enter para siguiente imagen, q para salir: ")

    if key.lower() == "q":
        break