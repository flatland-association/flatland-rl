# FAQ about the Flatland Challenge

These are the most common questions regarding the [Flatland Challenge](https://www.aicrowd.com/challenges/flatland-challenge).
If your questions are not answered please check the [Forum](https://discourse.aicrowd.com/c/flatland-challenge?_ga=2.33753761.1627822449.1571622829-1432296534.1549103074) and post your question there.

### How can I win prizes in this challenge?
You can win prizes in different categories.

Best Solution Prize: Won by the participants with the best performing submission on our test set. Only your rankings from the Round 1 and Round 2 are taken into account. Check the leader board on this site regularly for the latest information on your ranking.

The top three submissions in this category will be awarded the following cash prizes (in Swiss Francs):

- CHF 7'500.- for first prize
- CHF 5'000.- for second prize
- CHF 2'500.- for third prize

Community Contributions Prize: Awarded to the person/group who makes the biggest contribution to the community - done through generating new observations and sharing them with the community.

The top submission in this category will be awarded the following cash prize (in Swiss Francs): CHF 5'000.-

In addition, we will hand-pick and award up to five (5) travel grants (up to 1'500 CHF each)to the Applied Machine Learning Days 2019 in Lausanne, Switzerland. Participants with promising solutions may be invited to present their solutions at SBB in Bern, Switzerland.

To check your eligibility please read the prizes section in the [rules](https://www.aicrowd.com/challenges/flatland-challenge/challenge_rules/68).
### What are the deadlines for the flatland challenge?
- The beta round starts on the 1st of July 2019 and ends on the 30th of July 2019
- Round 1 closed on Sunday, 13th of October 2019, 12 PM. UTC +1
- Round 2 closes on Sunday, 5th of January 2020, 12 PM. UTC +1

### How is the score of a submission computed?
The scores of your submission are computed as follows:

1. Mean number of agents done, in other words how many agents reached their target in time.
2. Mean reward is just the mean of the cummulated reward.
3. If multiple participants have the same number of done agents we compute a "nomralized" reward as follows:
```
normalized_reward =cumulative_reward / (self.env._max_episode_steps +self.env.get_num_agents()
```
The mean number of agents done is the primary score value, only when it is tied to we use the "normalized" reward to determine the position on the leaderboard.

### How do I submit to the Flatland Challenge?
Follow the instructions in the [starter kit](https://github.com/AIcrowd/flatland-challenge-starter-kit) to get your first submission.

### Can I use env variables with my controller?
Yes you can. You can access all environment variables as you please. We recommend you use a custom observation builder to do so as explained [here](http://flatland-rl-docs.s3-website.eu-central-1.amazonaws.com/03_tutorials.html#custom-observations-and-custom-predictors-tutorial).

### What are the time limits for my submission?
If there is no action on the server for 10 minutes the submission will be cancelled and a time-out error wil be produced.

If the submissions in total takes longer than 8 hours a time-out will occur.

### What are the parameters for the environments for the submission scoring?
The environments vary in size and number of agents as well as malfunction parameters. The upper limit of these variables for submissions are:
- `(x_dim, y_dim) <= (150, 150)`
- `n_agents <= 250` (this might be updated)
- `malfunction rates` this is currently being refactored
