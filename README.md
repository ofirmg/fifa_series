# Football Insights from FIFA Data - Medium series
This repo contains the code base for the <a href="https://medium.com/@ofirmagdaci">"Football Insights from FIFA Data"</a> articles series by Ofir Magdaci at medium.com
The project consists of three papers:
1. <a href="https://medium.com/@ofirmagdaci/football-insights-from-fifa-data-what-comes-and-goes-with-age-2c4636bc99d1">What Comes and Goes with Age</a>
2. <a href="https://medium.com/@ofirmagdaci/football-insights-from-fifa-data-player-valuation-55b1b748e05d">Player Valuation</a>
3. <a href="https://medium.com/@ofirmagdaci/">Predicting the Future</a>

## Notes & disclaimers 
Two important disclaimers before you start:
1. The code of this repo is merely experimental. It was written in playground standrads and quality, without proper unittestings and any care for best practices. Do not use it for production purposes.
2. The dataset used in this project is not publicly available (at least not formally by EA). Hence, I will merely point to its sources (see below).
3. The last part code, 'Predicting the Future' is private and not part of this repository.

## Data 
The data for this project is essentially a mesh of three different datasets I found on Kaggle: the FIFA 21 Dataset link (stats from FIFA 15–21), the FIFA 22 dataset, and the FIFA 23 dataset. Combined, these datasets cover 45,630 players (by ID) and 1,017 clubs and teams over the years 2015–2023.

## Content
The repo includes 2 Jupyer notebooks, one with the code of chapter 1, and the other for part 2.
The other .py files holds utils functions to make the notebooks more readble.

## Required packages
- pandas
- scikit-learn
- plotly
- networkx

## License
As described in the license file, be my guest. 

## Contacts and communication:
- <a href="www.magdaci.com">Personal website</a>
- <a href="https://twitter.com/Magdaci">Twitter</a>
