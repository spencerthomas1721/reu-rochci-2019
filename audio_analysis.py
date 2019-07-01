import parselmouth as pm
import pandas as pd
import numpy as np
import glob

# operating under the assumption that i'll be the only person working with audio
data_path = "C:/Users/rltho/Desktop/REU/data/audio"
f_list = glob.glob(data_path + "/*")

# features correspond to the prosodic features measured in the mit experiment
# not including duration bc all videos were 2 minutes long
# not including jitter or shimmer due to parselmouth's limitations
features = ["file_name", "f0_mean", "f0_min", "f0_max", "f0_range", "f0_sd", "f1_mean", "f1_min", "f1_max", "f1_range", "f1_sd", "f1_bw",  "f2_mean", "f2_min", "f2_max", "f2_range", "f2_sd", "f2_bw", "f3_mean", "f3_min", "f3_max", "f3_range", "f3_sd", "f3_bw", "f2_f1_mean", "f2_f1_sd", "f3_f1_mean", "f3_f1_sd", "intensity_mean", "intensity_min", "intensity_max", "intensity_range", "intensity_sd", "percent_unvoiced", "percent_breaks", "max_pause", "mean_pause"]

snd_info = {i:[] for i in features}

for f in f_list:
    snd_info["file_name"].append(f.replace(data_path + "\\", ""))
    snd = pm.Sound(f)
    ptch = snd.to_pitch()
    mtrx = ptch.to_matrix()
    fmnt = snd.to_formant_burg()
    intensity = snd.to_intensity()

    formants = np.empty((3, mtrx.n_columns))
    formant_bandwidths = [0, 0, 0]

    intensity_list = []
    intensity_min = float("inf")
    intensity_max = 0

    pauses = []
    after_pause = False
    current_pause = [0, 0]
    pause_frames = 0

    for frame in range(1, mtrx.n_columns + 1):
        t = fmnt.get_time_from_frame_number(frame)
        for i in range(3):
            formants[i][frame-1] = fmnt.get_value_at_time(i+1, t)
            formant_bandwidths[i] += fmnt.get_bandwidth_at_time(i+1, t)
        t_intensity = intensity.get_value(t)
        intensity_list.append(t_intensity)
        if t_intensity < intensity_min:
            intensity_min = t_intensity
        elif t_intensity > intensity_max:
            intensity_max = t_intensity
        if t_intensity < 50:
            if after_pause:
                current_pause[1] = t
            else:
                current_pause[0] = t
            pause_frames += 1
            after_pause = True
        else:
            if after_pause:
                pauses.append(current_pause[1] - current_pause[0])
            after_pause = False
    snd_info["f0_min"].append(mtrx.get_minimum())
    snd_info["f0_max"].append(mtrx.get_maximum())
    snd_info["f0_mean"].append(np.sum(mtrx.values)/(mtrx.n_rows * mtrx.n_columns))
    snd_info["f0_range"].append(snd_info["f0_max"][-1] - snd_info["f0_min"][-1])
    snd_info["f0_sd"].append(np.std(np.array(mtrx)))

    snd_info["f1_mean"].append(formants[0].mean())
    snd_info["f1_min"].append(formants[0].min())
    snd_info["f1_max"].append(formants[0].max())
    snd_info["f1_range"].append(formants[0].max() - formants[0].min())
    snd_info["f1_sd"].append(np.std(formants[0]))
    snd_info["f1_bw"].append(formant_bandwidths[0]/mtrx.n_columns)
    snd_info["f2_mean"].append(formants[1].mean())
    snd_info["f2_min"].append(formants[1].min())
    snd_info["f2_max"].append(formants[1].max())
    snd_info["f2_range"].append(formants[1].max() - formants[1].min())
    snd_info["f2_sd"].append(np.std(formants[1]))
    snd_info["f2_bw"].append(formant_bandwidths[1]/mtrx.n_columns)
    snd_info["f3_mean"].append(formants[2].mean())
    snd_info["f3_min"].append(formants[2].min())
    snd_info["f3_max"].append(formants[2].max())
    snd_info["f3_range"].append(formants[2].max() - formants[2].min())
    snd_info["f3_sd"].append(np.std(formants[2]))
    snd_info["f3_bw"].append(formant_bandwidths[2]/mtrx.n_columns)
    f2_f1 = formants[1]/formants[0]

    snd_info["f2_f1_mean"].append(f2_f1.mean())
    snd_info["f2_f1_sd"].append(np.std(f2_f1))
    f3_f1 = formants[2]/formants[0]
    snd_info["f3_f1_mean"].append(f3_f1.mean())
    snd_info["f3_f1_sd"].append(np.std(f3_f1))

    snd_info["intensity_mean"].append(intensity.get_average())
    snd_info["intensity_min"].append(intensity_min)
    snd_info["intensity_max"].append(intensity_max)
    snd_info["intensity_range"].append(intensity_max - intensity_min)
    snd_info["intensity_sd"].append(np.nanstd(intensity_list))

    voiced_frames = ptch.count_voiced_frames()
    snd_info["percent_unvoiced"].append(1 - (voiced_frames/mtrx.n_columns))
    snd_info["percent_breaks"].append(pause_frames/mtrx.n_columns)
    snd_info["max_pause"].append(np.array(pauses).max())
    snd_info["mean_pause"].append(np.array(pauses).mean())

print(snd_info["file_name"])
#snd_df = pd.DataFrame(snd_info)
