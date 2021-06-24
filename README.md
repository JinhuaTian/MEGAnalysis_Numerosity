# MEGAnalysis_Numerosity
python script for MEG source reconstruction and decoding

CalRDM:
1. concatenate all data at subject level
2. down-sample data to 100 hz, slice data -0.1~0.7s
3. arrange data: picks=meg, normalize the data(mne.decoding.Scaler)
4. compute MEG RDM: pairwise SVM classification
Run RSA:
5. compute model RDM: number, field size, item size, low-level features
    arrange label
6. compute the partial Spearman correlation between MEG RDM and model RDM
