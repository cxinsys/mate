import os
import os.path as osp
import time
import datetime
from multiprocessing import Process, shared_memory, Semaphore

import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from jpype import *

class MATETENET(object):
    def solve(self,
              pairs=None,
              bin_arr=None,
              dt=1,
              ):
        if pairs is None:
            raise ValueError("pairs should be defined")

        if not dt:
            dt = 1

        entropy_final = []

        abs_path = osp.dirname(osp.abspath(__file__))

        jar_location = osp.join(abs_path, "infodynamics.jar")
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jar_location, "-Xmx16G")

        t_begin = time.time()

        for i, pair in enumerate(pairs):
            te_calc_class = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
            te_calc = te_calc_class()
            te_calc.setProperty("NORMALISE", "true")
            te_calc.initialise(dt, 0.5)

            te_calc.setObservations(JArray(JDouble, 1)(list(bin_arr[pair[1]])), JArray(JDouble, 1)(list(bin_arr[pair[0]])))

            entropy_final.append(te_calc.computeAverageLocalOfObservations())

            if (i % int(len(pairs) / 2)) == 0:
                print("{}: {}/{}".format(datetime.datetime.now(), i, len(pairs)))
            # end TE

        print("[TENET] Processing elapsed time: %f"%(time.time() - t_begin))

        return pairs, np.array(entropy_final)