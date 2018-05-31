# LoRa: Orthogonality in Chirp Spread Spectrum
Proving that orthogonality can allow for the demodulation of two orthogonal signals. This script evaluates the Symbol and Bit Error Rates (SER and BER) when demodulating a Chirp Spread Spectrum message with another orthogonal message as interfering.
Our goal is to compare, in theory, the level of interference we need to have in order to demodulate with good enough confidence in the orthogonal and non-orthogonal cases.

# Orthogonal signals

In Chirp Spread Spectrum, signal properties depend on two factors:
* The Spreading Factor (SF)
* The bandwidth
Orthogonal signals are signals such as $\frac{2^{SF_{1}}{B_{1}^{2}} \neq \frac{2^{SF_{2}}{B_{2}^{2}}$

# Test
In order to generate a SIR/SER and SIR/BER curve, move to the src directory and run one of the following command:
```
python main.py
python main.py -i 100 -p 150 -s 15 # for more thorough tests
```
To generate the same curve for non orthogonal signals, run
```
python main.py -no
```
Generating the curves for the default settings usually takes around 20 minutes.
