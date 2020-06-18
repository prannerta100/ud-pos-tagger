# ud-pos-tagger
Hidden Markov Model based POS tagging for 60+ languages on universal dependencies (UD) data

https://medium.com/@prannerta100/universal-dependencies-a-hidden-markov-quest-drem-yol-lok-2ca930ffc94f

### Introduction

In this notebook (HMM_UD.ipynb), we will use the [Pomegranate](http://pomegranate.readthedocs.io/en/latest/index.html) library to build a simple Hidden Markov Model for part-of-speech tagging.
 
The goal here is breadth rather than depth: we want to cover as many languages in the UD tagset as possible, therefore we did not implement additional features like:
- Laplace Smoothing [Wiki Link](https://en.wikipedia.org/wiki/Additive_smoothing) 
    
- Backoff Smoothing [Speech & Language Processing Ch. 4,9,10](https://web.stanford.edu/~jurafsky/slp3/) 

- Extending to Trigrams [Trigram Paper](http://www.coli.uni-saarland.de/~thorsten/publications/Brants-ANLP00.pdf)


આ પોથીમાં આપણે વિવિધ ભાષાઓમાં શબ્દ ભેદ (પાર્ટ્સ ઓફ સ્પીચ) ઉકેલવાનું કામ હિડેન માર્કોવ મોડેલ (HMM) વડે કરીશું. 

અહીં દાડમ લાઇબ્રેરી [Pomegranate](http://pomegranate.readthedocs.io/en/latest/index.html) વાપરવામાં આવી છે.

આપણું લક્ષ્ય ઊંડાણ ને બદલે વિસ્તારનો છે, એટલા માટે નીચે આપેલ ગુણવિશેષનો સમાવેશ નથી: 
- લાપ્લેસ નિયમિતકારણ [વિકિપીડિયા](https://en.wikipedia.org/wiki/Additive_smoothing) 
    
- બેકઓફ નિયમિતકરણ [ચોંપડી પાઠ 4,9,10](https://web.stanford.edu/~jurafsky/slp3/) 

- ટ્રાઇગ્રામ [સંશોધન પેપર](http://www.coli.uni-saarland.de/~thorsten/publications/Brants-ANLP00.pdf)


### Preprocessing and imports
We scan the entire UD folder to read in all the names of the respective language subdirectories, and prune out datasets that don't have train sets. Lack of a dev set is tolerated, as dev sets are fused to the training set, given the lack of iterative training in our HMM implementation. 

We need the following libraries installed:
1. Pomegranate
2. Numpy
3. Collections
4. pyconll

In addition, helper functions are found in data_prep.py and hmm_utils.py. Make sure you have these files in the same directory as this notebook!