#!/bin/bash
if [ ! -d ax650n_bsp_sdk ]; then
  echo "downloading ax650 bsp to ax650n_bsp_sdk, please wait..."
  wget -q https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/msp_50_3.10.2.zip -O msp_50_3.10.2.zip
  unzip -q msp_50_3.10.2.zip
  mv msp_50_3.10.2 ax650n_bsp_sdk
  rm -f msp_50_3.10.2.zip
fi

if [ ! -d ax620e_bsp_sdk ]; then
  echo "downloading ax620e bsp to ax620e_bsp_sdk, please wait..."
  wget -q https://github.com/ZHEQIUSHUI/assets/releases/download/ax_3.6.2/msp_20e_3.0.0.zip -O msp_20e_3.0.0.zip
  unzip -q msp_20e_3.0.0.zip
  mv msp_20e_3.0.0 ax620e_bsp_sdk
  rm -f msp_20e_3.0.0.zip
fi
