# LoRa: Orthogonality in Chirp Spread Spectrum
Proving that orthogonality can allow for the demodulation of two orthogonal signals. This script evaluates the Symbol and Bit Error Rates (SER and BER) when demodulating a Chirp Spread Spectrum message with another orthogonal message interfering.
Our goal is to compare, in theory, the maximum level of interference we can reach, in theory, in order to demodulate with good enough confidence in the orthogonal and non orthogonal cases.

# Orthogonal signals

In Chirp Spread Spectrum, signal properties depend on two factors:
* The Spreading Factor (SF)
* The Bandwidth (B)

The table below shows which pairs are orthogonal or not (x marks non orthogonal pairs)
![alt text](https://3.bp.blogspot.com/-qH-G97W5i7A/WHCFoWSzSRI/AAAAAAAAGsg/YE_foMWa6GQACRdwjXr9f-WDuBfCLDvLwCLcB/s1600/NonOrthogonal_Signals.png)

# Test
Install the requirements using
```
pip3 install -r requirements.txt
```
In order to generate a SIR/SER and SIR/BER curve, move to the src directory and run one of the following command:
```
python main.py
python main.py -i 100 -p 150 -s 15 # for longer simulations
```
To generate the same curve for non orthogonal signals, run
```
python main.py -no
```
Generating the curves for the default settings may take a few minutes.
