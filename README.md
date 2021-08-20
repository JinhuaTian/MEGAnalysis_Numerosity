#1. Preprocesing
   1.1 load data; run SSP; filter:notch, high pass, low pass; run ICA  # before epoch, 1 Hz highpass suggested; epoch data (auto rejection, baseline correction)
   1.2 Annotate and delete bad epochs manually
python script for MEG source reconstruction and decoding

CalRDM:
2.1 concatenate all data at subject level
2.2 down-sample data to 100 hz, slice data -0.1~0.7s
2.3 arrange data: picks=meg, normalize the data(mne.decoding.Scaler)
2.4 compute MEG RDM: pairwise SVM classification
Run RSA:
2.5 compute model RDM: number, field size, item size, low-level features
    arrange label
2.6 compute the partial Spearman correlation between MEG RDM and model RDM
