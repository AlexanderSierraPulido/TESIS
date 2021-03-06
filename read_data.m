%% Lectura de datos descargados de IRIS

%%%%%% Seismic event in Argentina on March 5th 2020  Mww5.5 %%%%%%

[hdrLCO_E, dataLCO_E] = load_sac('ARG_LCO_BH1_2020.sac'); % Header and data from Las Campanas Chile
[hdrLCO_N, dataLCO_N] = load_sac('ARG_LCO_BH2_2020.sac');
[hdrLCO_Z, dataLCO_Z] = load_sac('ARG_LCO_BHZ_2020.sac');
dataLCO_E = detrend(dataLCO_E);
dataLCO_N = detrend(dataLCO_N);
dataLCO_Z = detrend(dataLCO_Z);

[hdrLVZ_E, dataLVZ_E] = load_sac('ARG_LVZ_BH1_2020.sac'); % Header and data from Limon verde Chile
[hdrLVZ_N, dataLVZ_N] = load_sac('ARG_LVZ_BH2_2020.sac');
[hdrLVZ_Z, dataLVZ_Z] = load_sac('ARG_LVZ_BHZ_2020.sac');
dataLVZ_E = detrend(dataLVZ_E);
dataLVZ_N = detrend(dataLVZ_N);
dataLVZ_Z = detrend(dataLVZ_Z);

[hdrTRQA_E, dataTRQA_E] = load_sac('ARG_TRQA_BH1_2020.sac'); % Header and data from tornquist Argentina
[hdrTRQA_N, dataTRQA_N] = load_sac('ARG_TRQA_BH2_2020.sac');
[hdrTRQA_Z, dataTRQA_Z] = load_sac('ARG_TRQA_BHZ_2020.sac');
dataTRQA_E = detrend(dataTRQA_E);
dataTRQA_N = detrend(dataTRQA_N);
dataTRQA_Z = detrend(dataTRQA_Z);


save('ARG_LCO_E.mat','dataLCO_E')   % Save data into a mat file to be read in python
save('ARG_LCO_N.mat','dataLCO_N')
save('ARG_LCO_Z.mat','dataLCO_Z')

save('ARG_LVZ_E.mat','dataLVZ_E')
save('ARG_LVZ_N.mat','dataLVZ_N')
save('ARG_LVZ_Z.mat','dataLVZ_Z')

save('ARG_TRQA_E.mat','dataTRQA_E')
save('ARG_TRQA_N.mat','dataTRQA_N')
save('ARG_TRQA_Z.mat','dataTRQA_Z')

%%%%%% Seismic event in Albania on Sept 21st 2019 Mww5.6 %%%%%% 

[hdrANTO_E, dataANTO_E] = load_sac('ALB_ANTO_BH1_2019.sac'); % Header and data from Ankara Turkey
[hdrANTO_N, dataANTO_N] = load_sac('ALB_ANTO_BH2_2019.sac');
[hdrANTO_Z, dataANTO_Z] = load_sac('ALB_ANTO_BHZ_2019.sac');
dataANTO_E = detrend(dataANTO_E);
dataANTO_N = detrend(dataANTO_N);
dataANTO_Z = detrend(dataANTO_Z);

[hdrBFO_E, dataBFO_E] = load_sac('ALB_BFO_BHE_2019.sac'); % Header and data from Black Forest Observatory Germany
[hdrBFO_N, dataBFO_N] = load_sac('ALB_BFO_BHN_2019.sac');
[hdrBFO_Z, dataBFO_Z] = load_sac('ALB_BFO_BHZ_2019.sac');
dataBFO_E = detrend(dataBFO_E);
dataBFO_N = detrend(dataBFO_N);
dataBFO_Z = detrend(dataBFO_Z);

[hdrGRFO_E, dataGRFO_E] = load_sac('ALB_GRFO_BH1_2019.sac'); % Header and data from Grafenberg Germany
[hdrGRFO_N, dataGRFO_N] = load_sac('ALB_GRFO_BH2_2019.sac');
[hdrGRFO_Z, dataGRFO_Z] = load_sac('ALB_GRFO_BHZ_2019.sac');
dataGRFO_E = detrend(dataGRFO_E);
dataGRFO_N = detrend(dataGRFO_N);
dataGRFO_Z = detrend(dataGRFO_Z);

save('ARG_ANTO_E.mat','dataANTO_E')   % Save data into a mat file to be read in python
save('ARG_ANTO_N.mat','dataANTO_N')
save('ARG_ANTO_Z.mat','dataANTO_Z')

save('ARG_BFO_E.mat','dataBFO_E')   
save('ARG_BFO_N.mat','dataBFO_N')
save('ARG_BFO_Z.mat','dataBFO_Z')

save('ARG_GRFO_E.mat','dataGRFO_E')  
save('ARG_GRFO_N.mat','dataGRFO_N')
save('ARG_GRFO_Z.mat','dataGRFO_Z')

