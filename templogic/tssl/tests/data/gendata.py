import pickle
import os

import numpy as np

from templogic.tssl import inference

SHAPE = (16, 16)
MAXSPIRALS = 5
NSPIRALIMGS = 1000
NRANDOMIMGS = 1000
FN = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spirals.data")


def main():
    spirals = np.array(
        [inference.build_spirals(SHAPE, MAXSPIRALS) for i in range(NSPIRALIMGS)]
    )
    nonspirals = np.random.binomial(1, 0.5, (NRANDOMIMGS,) + SHAPE + (1,))
    data = {
        "imgs": np.vstack([spirals, nonspirals]),
        "labels": ([1] * NSPIRALIMGS) + ([0] * NRANDOMIMGS),
    }
    with open(FN, "wb") as f:
        pickle.dump(data, f)
        print(f"Written {len(data['labels'])} images")


if __name__ == "__main__":
    main()
