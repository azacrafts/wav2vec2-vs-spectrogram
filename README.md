# Generalist vs. Specialist: Wav2Vec2 vs. spectrogram models

CSCI 494 course project. We compare MFCC + Random Forest, a log-mel CNN, ResNet-18 (ImageNet) and Wav2Vec2 (speech pretrained, raw waveform) on ESC-10 (environmental sounds, mismatched domain for Wav2Vec2) and Speech Commands (speech, matched domain). We look at clean accuracy, cost, and accuracy and calibration (ECE) under white and real noise at 20, 10 and 0 dB SNR. `sc_small` is Speech Commands with only 320 training clips, to match the size of ESC-10.

### Results

Clean accuracy:

| | ESC-10 | Speech Commands | SC, 320 clips |
|---|---|---|---|
| RF | 0.782 | 0.633 | 0.416 |
| CNN | 0.832 | 0.970 | 0.598 |
| ResNet-18 | 0.872 | 0.980 | 0.494 |
| Wav2Vec2 | 0.822 | 0.984 | 0.950 |

Wav2Vec2 is not better than ResNet on ESC-10, and it is only clearly better with little speech data. It has 94M parameters (CNN: 0.24M) and takes about 94 min to train on the full Speech Commands (CNN: 8 min). Under strong noise all neural networks become overconfident on ESC-10 (ECE 0.5-0.8) but stay calibrated on Speech Commands (ECE below 0.1). On `sc_small` ResNet and CNN are overconfident under noise and Wav2Vec2 is not. Temperature scaling fitted on clean validation data does not fix this.

### Conclusion

Ребята, прочитайте пж вот это.

Wav2Vec2 turned out not to be a universal model. On environmental sounds (ESC-10) it is no better than a plain ResNet-18, and on speech it is clearly better only when there is little data (320 clips: 0.950 against at most 0.598 for the other models). With the full dataset its advantage almost disappears (0.984 against 0.970 for the CNN), while it costs 390 times more parameters and about 11 times more training time, so it is only worth it when data is scarce and the domain matches its pretraining. Under noise accuracy drops for every model, and on ESC-10 all neural networks become confidently wrong (ECE 0.5-0.8), while on Speech Commands they stay well calibrated. This is not unique to Wav2Vec2, and the `sc_small` control shows that data size alone does not explain it, since Wav2Vec2 stays calibrated there and ResNet and CNN do not. The results agree with the idea that a domain mismatch breaks calibration under noise, but we cannot prove it: this is one seed, and the two datasets also differ in clip length and noise source.

### Look at the results

The finished results are in `results/`, so there is no need to train anything. Open `notebooks/results.ipynb` (tables and plots). `notebooks/eda.ipynb` shows the data.

- `results/results.csv`: one row per dataset, model and noise condition (acc, f1, ece, nll, overconf, ece_ts, temperature)
- `results/reliability.csv`: confidence bins for reliability diagrams
- `results/cost.csv`: parameters, training time and latency per run

### Reproduce

Put ESC-50 in `data/ESC-50-master/` and Speech Commands v0.02 in `data/speech_commands_v0.02/` (the data is not in git), then:

    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
    make train

`make train` takes a few hours (Wav2Vec2 on Speech Commands is most of it). Finished experiments are skipped, so it can be stopped and restarted. `make run` executes the results notebook.

### Code

- `src/data.py`: loading, splits, noise, log-mel
- `src/models.py`: the four models and the training loop
- `src/metrics.py`: accuracy, F1, ECE, NLL, temperature scaling
- `src/main.py`: runs all experiments and writes `results/`

ESC-10 uses 5 folds (test fold, next fold for validation, rest for training), Speech Commands uses the official lists. Models are trained on clean audio only, noise is added to the test clips, and temperature is fitted on clean validation data.

### Limitations

One seed per experiment, and ESC-10 has only 400 test clips, so differences of a few points are not reliable. ESC-10 and Speech Commands differ in more than domain (clip length, class types, and the real noise comes from the Speech Commands background folder), so the results agree with the domain hypothesis but do not prove it.
