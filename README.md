# LoRa: Orthogonality in Chirp Spread Spectrum
Proving that orthogonality can allow for demodulating two orthogonal signals and outputting SIR/SER SIR/BER curves.

# Test
In order to generate a SIR/SER and SIR/BER curve, move to the src directory and run one of the following command:
'''
python main.py
python main.py -i 100 -p 150 -s 15 # for more thorough tests
'''
To generate the same curve for non orthogonal signals, run
'''
python main.py -no
'''
