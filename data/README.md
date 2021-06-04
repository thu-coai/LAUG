# Data
The data used in our paper are available at [Link](http://115.182.62.174:9876/) . Please download it and place it  `data/` dir.

Our data contains 2 datasets: MultiWOZ and Frames, along with their augmented copies.


## MultiWOZ
- Original data
  - We use MultiWOZ 2.3 as the original data. We place it at `data/multiwoz/` dir.
  - Train/val/test size: 8434/999/1000 dialogs.
  - LICENSE:
- Augmented data
  - We have 4 augmented testsets : 
    - WP (Word Perturbation), size: 1000, placed at `data/multiwoz/WP`.
    - TP (Text Paraphrasing), size: 1000, placed at `data/multiwoz/TP`.
    - SR (Speech Perturbation), size: 1000, placed at `data/multiwoz/SR`.
    - SD (Speech Disfluency), size: 1000, placed at `data/multiwoz/SD`.
  - We have 1 augmented training set :
    - Size : 16868 , Contains : 50%Original+(12.5%WP+12.5%TP+12.5%SR+12.5%SD) , placed at `data/multiwoz/Enhanced`.
- Real user evaluation data :
  - We collected 240 utterance from real users for our real user evaluation.
  - We place it at `data/multiwoz/Real` dir.
  - Please see our paper for detailed information about the statistics and collection of the real data.

## Frames
- Original data
  - We proccess Frames into the same format as MultiWOZ and place it at `data/Frames/` dir.
  - Train/val/test size: 1095/137/137 dialogs.
  - LICENSE:
- Augmented data
  - We have 4 augmented testsets : 
    - WP (Word Perturbation), size: 137, placed at `data/Frames/WP`.
    - TP (Text Paraphrasing), size: 137, placed at `data/Frames/TP`.
    - SR (Speech Perturbation), size: 137, placed at `data/Frames/SR`.
    - SD (Speech Disfluency), size: 137, placed at `data/Frames/SD`.
  - We have 1 augmented training set :
    - Size : 2190 , Contains : 50%Original+(12.5%WP+12.5%TP+12.5%SR+12.5%SD) , placed at `data/Frames/Enhanced`.
