# CNN, RNN, and LSTM Inference From Scratch

## Deskripsi Singkat
### Struktur Repository
```
├───data // data dari NusaX
│       test.csv
│       train.csv
│       valid.csv
└───src
    │   utils.py
    ├───cnn
    │       activation.py
    │       cnn.py
    │       cnn_training.ipynb
    │       conv_layer.py
    │       dense_layer.py
    │       pooling.py
    ├───lstm
    │       lstm.py
    │       lstm_training.ipynb
    └───rnn
            rnn.py
            rnn_training.ipynb
```
- Folder `data` berisi dengan data dari NusaX yang digunakan dalam training RNN dan LSTM
- Folder `src` berisi source code repository
    - Folder `cnn`, `rnn`, `lstm` berisi file implementasi masing-masing neural network tersebut from scratch, serta hasil pelatihan dengan variasi hyperparameter.
    - Training hyperparameter dapat ditemukan dalam file `{nama-model}_training.ipynb`
## Cara Setup dan Run
### CNN
#### From Scratch Inference
- Jalankan `py src/cnn/cnn.py`
#### Hyperparameter Training
- Buka file cnn_training.ipynb
- Jalankan Jupyter Notebook
### RNN
- Buka file rnn_training.ipynb
- Jalankan Jupyter Notebook
### LSTM
- Buka file lstm_training.ipynb
- Jalankan Jupyter Notebook
## Pembagian Tugas
| NIM | Nama | Tugas |
| - | - | - |
| 13522102 | Hayya Zuhailii Kinasih | RNN, LSTM, Laporan |
| 13522104 | Diana Tri Handayani | CNN, Laporan |
