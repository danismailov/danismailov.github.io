+++
title = 'Teaching learning machines football pt.3'
date = 2024-03-20T12:51:48Z
draft = true
+++


### *First day on the training ground: our first shot at training FPL's very own mystic meg using machine learning*

*See the Github repo [here](https://github.com/danismailov/fpl-with-fml) and the previous blog posts [here](/posts/teaching-learning-machines-football-pt2) and [here](/posts/teaching-learning-machines-football).*




# `TODO`




## How do we know which features are helpful?

Helpfully for us, there's known approaches to ***interpretability***, or our ability to know what's going on under the hood of the model. 

In this case, once our model is trained in the next post, there's a function from AutoGluon called `predictor.feature_importance()` which we'll use to better understand the specific importance of each feature.


### *How do we know how useful each stat could be?*
You might already have some inklings about a few of the FPL stats above. For example:

- ***Transfers*** (both in or out) might be a great proxy for public "hunches" about players who are outperforming or underperforming their usual form, which isn't always be obvious from the data.

- On the flip side, `penalties_missed` isn't very indicative of relative form, as most players across most games wouldn't even dream of taking a penalty, meaning this stat would be a desperately boring 0.

Luckily, there's a scientific way to interpret exactly how much predictive power each feature has, which I'll come onto in the next post (yes I'm hoping that [permutation importance](https://explained.ai/rf-importance/) works as a nerdy hook for the next instalment in the series..). 





### Missing values

Using the following snippet for the `count_na` function, which counts the number of missing values across each of the 42 columns, we can also see how missing values are distributed amongst the columns:

&nbsp;


```python
def count_na(data):
    na_count = data.isna().sum()
    with pd.option_context('display.max_rows', None,
            'display.max_columns', None,
            'display.precision', 3):
        print(na_count)

count_na(data=gw_rows)
```

Output:

```python
# Column name                       # Missing values
name                                               0
assists                                            0
bonus                                              0
bps                                                0
clean_sheets                                       0
creativity                                         0
element                                            0
fixture                                            0
goals_conceded                                     0
goals_scored                                       0
ict_index                                          0
influence                                          0
kickoff_time                                       0
minutes                                            0
opponent_team                                      0
own_goals                                          0
penalties_missed                                   0
penalties_saved                                    0
red_cards                                          0
round                                              0
saves                                              0
selected                                           0
team_a_score                                      59
team_h_score                                      59
threat                                             0
total_points                                       0
transfers_balance                                  0
transfers_in                                       0
transfers_out                                      0
value                                              0
was_home                                           0
yellow_cards                                       0
opponent                                           0
season                                             0
position                                       16556
team                                           16556
xP                                             16556
expected_assists                               66368
expected_goal_involvements                     66368
expected_goals                                 66368
expected_goals_conceded                        66368
starts                                         66368
```



# Next blog post – Training our first model on FPL data

- Hyperparameter tuning
- Further tweaking of the training data - e.g. filtering number of players to train on ones that play
    - *[Violin Plot]*
    While it's some complicated shenanigans that underpins this ([see here](https://explained.ai/rf-importance/) for a more technical explanation of permutation importance), this effectively shuffles the values of one feature, to see how this affects the accuracy of the model.