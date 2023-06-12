#!/bin/bash
#DATASET="$1"
EXP_CMD="python src/inference.py experiment=inf_resgated_multidigraph"

BASELINE1="models.net.num_layers=1 models.net.hidden_features=64"
BASELINE2="models.net.num_layers=1 models.net.hidden_features=32"
BASELINE3="models.net.num_layers=1 models.net.hidden_features=16"
BASELINE4="models.net.num_layers=2 models.net.hidden_features=64"
BASELINE5="models.net.num_layers=2 models.net.hidden_features=32"
BASELINE6="models.net.num_layers=2 models.net.hidden_features=16"
BASELINE7="models.net.num_layers=3 models.net.hidden_features=16"
BASELINE8="models.net.num_layers=3 models.net.hidden_features=32"


for BASELINE in "$BASELINE1" "$BASELINE2" "$BASELINE3" "$BASELINE4" "$BASELINE5" "$BASELINE6" "$BASELINE7" "$BASELINE8"
do
    echo "Baseline is $BASELINE"

    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_22_real_06_07
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_21_real_06_06
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_20_real_05_01
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_19_real_04_29
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_18_real_04_29
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_17_real_05_01
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_16_real_05_01
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_15_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_14_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_13_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_12_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_11_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_10_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_9_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_8_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_7_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_6_real_06_06
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_5_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_4_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_3_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_2_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_1_real_04_30
    PROJECT_ROOT="./" $EXP_CMD $BASELINE dataset_name=chr_X_real_05_01
done
