import os
os.environ["OMP_NUM_THREADS"] = "1"        # must run BEFORE `import numpy`
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np