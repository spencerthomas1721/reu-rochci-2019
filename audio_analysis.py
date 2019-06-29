import parselmouth as pm
import pandas as pd
import numpy as np
import glob

# note: i know this is a mess, will clean up later

# operating under the assumption that i'll be the only person working with audio
data_path = "C:/Users/rltho/Desktop/REU/data/audio/*"
f_list = glob.glob(data_path)

# columns correspond to the prosodic features measured in the mit experiment
# will be columns of pandas dataframe
# not including duration bc all videos were 2 minutes long
columns = ["file_name", "f0_mean", "f0_min", "f0_max", "f0_range", "f0_sd", "f1_mean", "f1_min", "f1_max", "f1_range", "f1_sd", "f1_bw",  "f2_mean", "f2_min", "f2_max", "f2_range", "f2_sd", "f2_bw", "f3_mean", "f3_min", "f3_max", "f3_range", "f3_sd", "f3_bw", "f2_f1_mean", "f2_f1_sd", "f3_f1_mean", "f3_f1_sd", "intensity_mean", "intensity_min", "intensity_max", "intensity_range", "intensity_sd", "jitter", "shimmer", "percent_unvoiced", "percent_breaks", "max_pause", "mean_pause"]
snd_info = {i:[] for i in columns}
snd_info["file_name"] = f_list

for f in f_list:
    snd = pm.Sound(f)
    ptch = snd.to_pitch()
    mtrx = ptch.to_matrix()
    fmnt = snd.to_formant_burg()
    #print(ptch.tmin)
    print(ptch.ts)
    #print(ptch.to_array())
    #print(ptch.__getitem__(1))
    formants = np.empty((3, mtrx.n_columns))
    #min = [float("inf"), float("inf"), float("inf")]
    for frame in range(mtrx.n_columns):
        t = fmnt.get_time_from_frame_number(frame)
        for i in range(3):
            formants[i][frame] = fmnt.get_value_at_time(i + 1, t)
    snd_info["f1_mean"].append(formants[0].mean())
    snd_info["f1_min"].append(formants[0].min())
    snd_info["f1_max"].append(formants[0].max())
    snd_info["f1_range"].append(formants[0].max() - formants[0].min())
    snd_info["f1_sd"].append(np.std(formants[0]))
    snd_info["f2_mean"].append(formants[1].mean())
    snd_info["f2_min"].append(formants[1].min())
    snd_info["f2_max"].append(formants[1].max())
    snd_info["f2_range"].append(formants[1].max() - formants[1].min())
    snd_info["f2_sd"].append(np.std(formants[1]))
    snd_info["f3_mean"].append(formants[2].mean())
    snd_info["f3_min"].append(formants[2].min())
    snd_info["f3_max"].append(formants[2].max())
    snd_info["f3_range"].append(formants[2].max() - formants[2].min())
    snd_info["f1_sd"].append(np.std(formants[2]))
    snd_info["f2_f1_mean"].append(snd_info["f2_mean"][-1]/snd_info["f1_mean"][-1])
    snd_info["f3_f1_mean"].append(snd_info["f3_mean"][-1]/snd_info["f1_mean"][-1])

    snd_info["f0_min"].append(mtrx.get_minimum())
    snd_info["f0_max"].append(mtrx.get_maximum())
    snd_info["f0_mean"].append(np.sum(mtrx.values)/(mtrx.n_rows * mtrx.n_columns))
    snd_info["f0_range"].append(snd_info["f0_max"][-1] - snd_info["f0_min"][-1])
    voiced_frames = ptch.count_voiced_frames()
    snd_info["intensity_mean"].append(snd.to_intensity().get_average())
    snd_info["percent_unvoiced"].append(1 - (voiced_frames/mtrx.n_rows))
    # find a way to get jitter, shimmer?
    # find a way to clip off ends to get accurate pauses/breaks/intensity?

#snd_df = pd.DataFrame(snd_info)
