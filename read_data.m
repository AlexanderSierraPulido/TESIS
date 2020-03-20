%% prueba de lectura de datos 

%%%%%% Seismic event on Argentina on March 5th 2020  Mww5.5 %%%%%%

[hdrLCO_E, dataLCO_E] = load_sac('ARG_LCO_BH1_2020.sac'); % Header and data from Las Campanas Chile
[hdrLCO_N, dataLCO_N] = load_sac('ARG_LCO_BH2_2020.sac');
[hdrLCO_Z, dataLCO_Z] = load_sac('ARG_LCO_BHZ_2020.sac');

[hdrLVZ_E, dataLVZ_E] = load_sac('ARG_LVZ_BH1_2020.sac'); % Header and data from Limon verde Chile
[hdrLVZ_N, dataLVZ_N] = load_sac('ARG_LVZ_BH2_2020.sac');
[hdrLVZ_Z, dataLVZ_Z] = load_sac('ARG_LVZ_BHZ_2020.sac');

[hdrTRQA_E, dataTRQA_E] = load_sac('ARG_TRQA_BH1_2020.sac'); % Header and data from tornquist Argentina
[hdrTRQA_N, dataTRQA_N] = load_sac('ARG_TRQA_BH2_2020.sac');
[hdrTRQA_Z, dataTRQA_Z] = load_sac('ARG_TRQA_BHZ_2020.sac');


save('ARG_LCO.mat','dataLCO_E')

%%%%%% Seismic event on Albania on Sept 21st 2019 Mww5.6 %%%%%% 

[hdrANTO_E, dataANTO_E] = load_sac('ALB_ANTO_BH1_2019.sac'); % Header and data from Ankara Turkey
[hdrANTO_N, dataANTO_N] = load_sac('ALB_ANTO_BH2_2019.sac');
[hdrANTO_Z, dataANTO_Z] = load_sac('ALB_ANTO_BHZ_2019.sac');

[hdrBFO_E, dataBFO_E] = load_sac('ALB_BFO_BHE_2019.sac'); % Header and data from Black Forest Observatory Germany
[hdrBFO_N, dataBFO_N] = load_sac('ALB_BFO_BHN_2019.sac');
[hdrBFO_Z, dataBFO_Z] = load_sac('ALB_BFO_BHZ_2019.sac');

[hdrGRFO_E, dataGRFO_E] = load_sac('ALB_GRFO_BH1_2019.sac'); % Header and data from Grafenberg Germany
[hdrGRFO_N, dataGRFO_N] = load_sac('ALB_GRFO_BH2_2019.sac');
[hdrGRFO_Z, dataGRFO_Z] = load_sac('ALB_GRFO_BHZ_2019.sac');
