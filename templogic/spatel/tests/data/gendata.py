import pickle
import os
import itertools as it

import numpy as np

from templogic.tssl import inference

SHAPE = (8, 32, 32)
MAXSPIRALS = 15
SPIRALSIZE = 3
NSPIRALIMGS = 1000
NRANDOMIMGS = 1000
FN = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spirals.data")


def main():
    spirals = np.array(
        [
            inference.build_rotating_spirals(SHAPE, MAXSPIRALS, SPIRALSIZE)
            for i in range(NSPIRALIMGS)
        ]
    )
    nonspirals = np.random.binomial(1, 0.5, (NRANDOMIMGS,) + SHAPE + (1,))
    signals = [
        list(zip(it.cycle(range(SHAPE[0])), movie))
        for movie in np.vstack([spirals, nonspirals])
    ]
    data = {
        "signals": signals,
        "labels": ([1] * NSPIRALIMGS) + ([0] * NRANDOMIMGS),
        "depth": int(np.sqrt(SHAPE[0])),
    }
    with open(FN, "wb") as f:
        pickle.dump(data, f)
        print(f"Written {len(data['labels'])} images")


if __name__ == "__main__":
    main()
